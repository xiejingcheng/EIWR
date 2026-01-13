import time 
import heapq 
import os
import gc
import torch 
import torch.nn as nn 
import transformers
import numpy as np
from tqdm import tqdm
import logging

from typing import List, Optional, Tuple, Union
from torch import nn

from .sparsegpt import SparseGPT, SparseGPT_NoReconstruct, SparseGPTV3, SparseGPTV4
from .layerwrapper import WrappedGPT, WrappedGPTV3, WrappedGPTV4, WrappedGPTV6, WrappedGPTV10
from .data import get_loaders, prepare_calibration_input

from .ablate import AblateGPT 

# from .matmul_had import *
from .utils import *

debug = True


valid_out = True

def return_reorder_indice(input_tensor):
    """
    For instance:
    [[1., -2., 3.],
    [-2, 2., -4],
    [5., 6., -7],
    [-6, -7, -4]]
    return indices of
    [[-2.,  3.,  1.],
    [-2., -4.,  2.],
    [-7.,  6.,  5.],
    [-6., -7., -4.]]
    Description: The relative order in the positive number remains unchanged, and the relative order in the negative number is flipped.
    """
    positive_tensor = input_tensor.clone()
    negative_tensor = input_tensor.clone()

    positive_mask = positive_tensor > 0
    negative_mask = negative_tensor < 0

    positive_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )
    negative_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )

    positive_indices[~positive_mask] = float("inf")
    negative_indices[~negative_mask] = float("inf")

    positive_value, _ = torch.sort(positive_indices, dim=1)
    negative_value, _ = torch.sort(negative_indices, dim=1)

    positive_value = torch.flip(positive_value, dims=[1])

    negative_value[negative_value == float("inf")] = 0
    positive_value[positive_value == float("inf")] = 0

    reorder_indice = (positive_value + negative_value).to(torch.int64)

    return reorder_indice

@torch.no_grad()
def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            if dual_ascent:
                wrapped_layers[name] = WrappedGPTV3(subset[name], valid=valid_out)
            else:
                wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()  

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
                    k+=1
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    wrapped_layers[name].validate()

                flag, alpha, beta = wrapped_layers[name].get_args()
                min_iter = 0
                if flag:
                    wrapped_layers[name].dual_ascent2(beta = beta, alpha = alpha, min_iter=min_iter, theld= args.dual_theld)
                    if debug:
                        print("Validation after dual ascent:")
                        wrapped_layers[name].validate()
                    wrapped_layers[name].del_valid()
                else:
                    del wrapped_layers[name].H, wrapped_layers[name].H_B
                    torch.cuda.empty_cache()

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()   

        inps, outs = outs, inps
        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, device, prune_n=0, prune_m=0, no_reconstruct=False, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):  
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)    

        gpts = {}
        for name in subset:
            if dual_ascent:
                gpts[name] = SparseGPTV3(subset[name], valid=valid_out)
            else:
                gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            gpts[name].fasterprune(ratios[k], prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    gpts[name].validate()
                
                flag, alpha, beta = gpts[name].get_args()
                min_iter = 0
                
                if ratios[k] < args.dual_sp_theld:
                    flag = False

                if flag:
                    gpts[name].dual_ascent2(beta = beta, alpha = alpha, min_iter=min_iter, theld= args.dual_theld)
                    if debug:
                        print("Validation after dual ascent:")
                        gpts[name].validate()
                    gpts[name].del_valid()
                else:
                    del gpts[name].H, gpts[name].H_B
                    torch.cuda.empty_cache()

            gpts[name].free()
            k+=1

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()

        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda
        torch.cuda.empty_cache()
        # if debug:
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/inps_layer{i}.npy", inps.cpu().numpy())
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/outs_layer{i}.npy", outs.cpu().numpy())
        print(f"layer {i} done")
        # print("Inps max and mean: ", outs.max(), outs.abs().mean())

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_magnitude(args, model, tokenizer, device, prune_n=0, prune_m=0, no_reconstruct=False, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):
    if dual_ascent:
        use_cache = model.config.use_cache 
        model.config.use_cache = False 
        dev = cuda_device

        print("loading calibdation data")
        dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
        print("dataset loading complete")
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

        layers = model.model.layers

        layer_num = len(find_layers(layers))
        if ratios is None:
            ratios = [args.sparsity_ratio for i in range(layer_num)]
        k=0

        for i in range(len(layers)):
            layer_cuda = layers[i].to(dev)
            if exclude is not None:
                subset = find_layers_without(layer_cuda, exclude=exclude)
            else:
                subset = find_layers(layer_cuda)

            try:    
                dev = model.hf_device_map[f"model.layers.{i}"]
            except:
                dev = dev
            if attention_mask is None:
                position_ids = position_ids.to(dev)
            else:
                attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)

            wrapped_layers = {}
            for name in subset:
                if dual_ascent:
                    wrapped_layers[name] = WrappedGPTV3(subset[name], valid=valid_out)
                else:
                    wrapped_layers[name] = WrappedGPT(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            batch_num = int(args.nsamples / args.n_batch)

            for b in range(batch_num):
                i1 = b * args.n_batch
                i2 = (b + 1) * args.n_batch

                inps_cuda = inps[i1:i2].to(dev)
                outs_cuda = torch.zeros_like(inps_cuda)

                for j in range(args.n_batch):
                    if attention_mask is None:
                        outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                    else:
                        outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

                outs_cuda = outs_cuda.cpu()
                inps_cuda = inps_cuda.cpu()
                outs[i1:i2] = outs_cuda
                torch.cuda.empty_cache()  

            for h in handles:
                h.remove()

            for name in subset:
                print(f"pruning layer {i} name {name}")
                W_metric = torch.abs(subset[name].weight.data)

                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # wanda variant 
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha

                            alpha = alpha_new 
                            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # unstructured pruning
                        indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
                        k+=1
                        W_mask.scatter_(1, indices, True)

                subset[name].weight.data[W_mask] = 0  ## set weights to zero 

                if dual_ascent:
                    if debug:
                        print("Validation after prune:")
                        wrapped_layers[name].validate()

                    flag, alpha, beta = wrapped_layers[name].get_args()
                    min_iter = 0
                    if flag:
                        wrapped_layers[name].dual_ascent4(beta = beta, alpha = alpha, epsilon=args.epsilon, min_iter=min_iter, theld= args.dual_theld)
                        if debug:
                            print("Validation after dual ascent:")
                            wrapped_layers[name].validate()
                        wrapped_layers[name].del_valid()
                    else:
                        del wrapped_layers[name].H, wrapped_layers[name].H_B
                        torch.cuda.empty_cache()

            for b in range(batch_num):
                i1 = b * args.n_batch
                i2 = (b + 1) * args.n_batch

                inps_cuda = inps[i1:i2].to(dev)
                outs_cuda = torch.zeros_like(inps_cuda)

                for j in range(args.n_batch):
                    if attention_mask is None:
                        outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                    else:
                        outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

                outs_cuda = outs_cuda.cpu()
                inps_cuda = inps_cuda.cpu()
                outs[i1:i2] = outs_cuda
                torch.cuda.empty_cache()   

            inps, outs = outs, inps
            layer_cuda = layer_cuda.cpu()
            layers[i] = layer_cuda

        model.config.use_cache = use_cache 
        torch.cuda.empty_cache()
    else:
        print("mp")
        layers = model.model.layers 

        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)

            for name in subset:
                W = subset[name].weight.data 
                W_metric = torch.abs(W)
                if prune_n != 0:
                    W_mask = (torch.zeros_like(W)==1)
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                else:
                    thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric<=thresh)

                W[W_mask] = 0
                subset[name].weight.data = W

@torch.no_grad()
def prune_sparsegpt_ww(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, dual_ascent=False, cuda_device="cuda"):
    s1 = 1.0 - args.ww_epsilon
    s2 = 1.0 + args.ww_epsilon

    all_layer_ratio = ww_sparsity(args, model, device, s1, s2)
    # sparsegpt pruning
    prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, ratios=all_layer_ratio, dual_ascent=dual_ascent, cuda_device=cuda_device)

@torch.no_grad()
def prune_wanda_ww(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, dual_ascent=False, cuda_device="cuda"):
    s1 = 1.0 - args.ww_epsilon
    s2 = 1.0 + args.ww_epsilon

    all_layer_ratio = ww_sparsity(args, model, device, s1, s2)
    # wanda pruning
    prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, ratios=all_layer_ratio, dual_ascent=dual_ascent, cuda_device=cuda_device)

@torch.no_grad()
def prune_wanda_2(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            if dual_ascent:
                wrapped_layers[name] = WrappedGPTV3(subset[name], valid=valid_out)
            else:
                wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()  

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
                    k+=1
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    wrapped_layers[name].validate()

                flag, alpha, beta = wrapped_layers[name].get_args()
                min_iter = 0
                if flag:
                    wrapped_layers[name].dual_ascent3(beta = beta, alpha = alpha, min_iter=min_iter, theld= args.dual_theld)
                    if debug:
                        print("Validation after dual ascent:")
                        wrapped_layers[name].validate()
                    wrapped_layers[name].del_valid()
                else:
                    del wrapped_layers[name].H, wrapped_layers[name].H_B

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()   

        inps, outs = outs, inps
        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_sparsegpt_eff(args, model, tokenizer, device, prune_n=0, prune_m=0, no_reconstruct=False, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):  
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0
    #os.system("nvidia-smi")
    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)
        print("load layer")    
        #os.system("nvidia-smi")
        gpts = {}
        for name in subset:
            if dual_ascent:
                gpts[name] = SparseGPTV4(subset[name], valid=valid_out)
            else:
                gpts[name] = SparseGPT(subset[name])
        print("load spgpt") 
        #os.system("nvidia-smi")
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()
        
        print("after forward")
        #os.system("nvidia-smi")

        for h in handles:
            h.remove()

        for name in gpts:
            #os.system("nvidia-smi")
            print(i, name)
            print('Pruning ...')
            if debug:
                gpts[name].validate()
            gpts[name].fasterprune(ratios[k], prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    gpts[name].validate()
                
                flag, alpha, beta = gpts[name].get_args()
                min_iter = 0
                
                if ratios[k] < args.dual_sp_theld:
                    flag = False

                if flag:
                    gpts[name].dual_ascent2(beta = beta, alpha = alpha, min_iter=min_iter, theld= args.dual_theld)
                    if debug:
                        print("Validation after dual ascent:")
                        gpts[name].validate()
                    gpts[name].del_valid()
                    torch.cuda.empty_cache()
                else:
                    del gpts[name].H, gpts[name].ora_W
                    torch.cuda.empty_cache()

            gpts[name].free()
            gpts[name] = None
            gc.collect()
            torch.cuda.empty_cache()
            k+=1

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()

        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda
        torch.cuda.empty_cache()
        # if debug:
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/inps_layer{i}.npy", inps.cpu().numpy())
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/outs_layer{i}.npy", outs.cpu().numpy())
        print(f"layer {i} done")
        # print("Inps max and mean: ", outs.max(), outs.abs().mean())

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_sparsegpt_eff_v2(args, model, tokenizer, device, prune_n=0, prune_m=0, no_reconstruct=False, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):  
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0
    #os.system("nvidia-smi")
    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)
        print("load layer")    
        #os.system("nvidia-smi")
        gpts = {}
        for name in subset:
            if dual_ascent:
                gpts[name] = SparseGPTV4(subset[name], valid=valid_out)
            else:
                gpts[name] = SparseGPT(subset[name])
        print("load spgpt") 
        #os.system("nvidia-smi")
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()
        
        print("after forward")
        #os.system("nvidia-smi")

        for h in handles:
            h.remove()

        for name in gpts:
            #os.system("nvidia-smi")
            print(i, name)
            print('Pruning ...')
            if debug:
                gpts[name].validate()
            gpts[name].fasterprune(ratios[k], prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    gpts[name].validate()
                
                flag, alpha, beta = gpts[name].get_args()
                min_iter = 0
                
                if ratios[k] < args.dual_sp_theld:
                    flag = False

                if flag:
                    gpts[name].dual_ascent6(beta = beta, alpha = alpha, min_iter=min_iter, theld= args.dual_theld)
                    if debug:
                        print("Validation after dual ascent:")
                        gpts[name].validate()
                    gpts[name].del_valid()
                    torch.cuda.empty_cache()
                else:
                    del gpts[name].H, gpts[name].ora_W
                    torch.cuda.empty_cache()

            gpts[name].free()
            gpts[name] = None
            gc.collect()
            torch.cuda.empty_cache()
            k+=1

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()

        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda
        torch.cuda.empty_cache()
        # if debug:
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/inps_layer{i}.npy", inps.cpu().numpy())
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/outs_layer{i}.npy", outs.cpu().numpy())
        print(f"layer {i} done")
        # print("Inps max and mean: ", outs.max(), outs.abs().mean())

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_sparsegpt_2(args, model, tokenizer, device, prune_n=0, prune_m=0, no_reconstruct=False, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):  
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)    

        gpts = {}
        for name in subset:
            if dual_ascent:
                gpts[name] = SparseGPTV3(subset[name], valid=valid_out)
            else:
                gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            gpts[name].fasterprune(ratios[k], prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    gpts[name].validate()
                
                flag, alpha, beta = gpts[name].get_args()
                min_iter = 0
                
                if ratios[k] < args.dual_sp_theld:
                    flag = False

                if flag:
                    gpts[name].dual_ascent3(beta = beta, alpha = alpha, min_iter=min_iter, theld= args.dual_theld)
                    if debug:
                        print("Validation after dual ascent:")
                        gpts[name].validate()
                    gpts[name].del_valid()
                else:
                    del gpts[name].H, gpts[name].H_B

            gpts[name].free()
            k+=1

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()

        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda
        torch.cuda.empty_cache()
        # if debug:
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/inps_layer{i}.npy", inps.cpu().numpy())
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/outs_layer{i}.npy", outs.cpu().numpy())
        print(f"layer {i} done")
        # print("Inps max and mean: ", outs.max(), outs.abs().mean())

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_sparsegpt_3(args, model, tokenizer, device, prune_n=0, prune_m=0, no_reconstruct=False, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):  
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)    

        gpts = {}
        for name in subset:
            if dual_ascent:
                gpts[name] = SparseGPTV3(subset[name], valid=valid_out)
            else:
                gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            gpts[name].fasterprune(ratios[k], prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    gpts[name].validate()
                
                flag, alpha, beta = gpts[name].get_args()
                min_iter = 0
                
                if ratios[k] < args.dual_sp_theld:
                    flag = False

                if flag:
                    gpts[name].dual_ascent4(beta = beta, alpha = alpha, epsilon=args.epsilon, min_iter=min_iter, theld= args.dual_theld)
                    if debug:
                        print("Validation after dual ascent:")
                        gpts[name].validate()
                    gpts[name].del_valid()
                else:
                    del gpts[name].H, gpts[name].H_B

            gpts[name].free()
            k+=1

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()

        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda
        torch.cuda.empty_cache()
        # if debug:
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/inps_layer{i}.npy", inps.cpu().numpy())
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/outs_layer{i}.npy", outs.cpu().numpy())
        print(f"layer {i} done")
        # print("Inps max and mean: ", outs.max(), outs.abs().mean())

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_wanda_3(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            if dual_ascent:
                wrapped_layers[name] = WrappedGPTV3(subset[name], valid=valid_out)
            else:
                wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()  

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
                    k+=1
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    wrapped_layers[name].validate()

                flag, alpha, beta = wrapped_layers[name].get_args()
                min_iter = 0
                if flag:
                    wrapped_layers[name].dual_ascent4(beta = beta, alpha = alpha, epsilon=args.epsilon, min_iter=min_iter, theld= args.dual_theld)
                    if debug:
                        print("Validation after dual ascent:")
                        wrapped_layers[name].validate()
                    wrapped_layers[name].del_valid()
                else:
                    del wrapped_layers[name].H, wrapped_layers[name].H_B
                    torch.cuda.empty_cache()

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()   

        inps, outs = outs, inps
        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_wanda_5(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            if dual_ascent:
                wrapped_layers[name] = WrappedGPTV6(subset[name], valid=valid_out, ora_W=True)
            else:
                wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()  

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
                    k+=1
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    wrapped_layers[name].validate()

                flag, alpha, beta = wrapped_layers[name].get_args()
                min_iter = 0
                if flag:
                    wrapped_layers[name].dual_ascent5(beta = beta, alpha = alpha, epsilon=args.epsilon, min_iter=min_iter, theld= args.dual_theld)
                    if debug:
                        print("Validation after dual ascent:")
                        wrapped_layers[name].validate()
                    wrapped_layers[name].del_valid()
                else:
                    del wrapped_layers[name].H, wrapped_layers[name].H_B
                    torch.cuda.empty_cache()

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()   

        inps, outs = outs, inps
        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_wanda_eff(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            if dual_ascent:
                wrapped_layers[name] = WrappedGPTV4(subset[name], valid=valid_out)
            else:
                wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()  

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
                    k+=1
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    wrapped_layers[name].validate()

                flag, alpha, beta = wrapped_layers[name].get_args()
                min_iter = 0
                if flag:
                    wrapped_layers[name].dual_ascent2(beta = beta, alpha = alpha, min_iter=min_iter, theld= args.dual_theld)
                    if debug:
                        print("Validation after dual ascent:")
                        wrapped_layers[name].validate()
                    wrapped_layers[name].del_valid()
                else:
                    del wrapped_layers[name].H, wrapped_layers[name].H_B
                    torch.cuda.empty_cache()

            wrapped_layers[name] = None
            gc.collect()
            torch.cuda.empty_cache()

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()   

        inps, outs = outs, inps
        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_wanda_eff_v2(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")

    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            if dual_ascent:
                wrapped_layers[name] = WrappedGPTV4(subset[name], valid=valid_out)
            else:
                wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()  

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
                    k+=1
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    wrapped_layers[name].validate()

                flag, alpha, beta = wrapped_layers[name].get_args()
                min_iter = 0
                if flag:
                    wrapped_layers[name].dual_ascent2(beta = beta, alpha = alpha, min_iter=min_iter, theld= args.dual_theld)
                    if debug:
                        print("Validation after dual ascent:")
                        wrapped_layers[name].validate()
                    wrapped_layers[name].del_valid()
                else:
                    del wrapped_layers[name].H, wrapped_layers[name].ora_W
                    torch.cuda.empty_cache()

            wrapped_layers[name] = None
            gc.collect()
            torch.cuda.empty_cache()

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()   

        inps, outs = outs, inps
        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_sparsegpt_4(args, model, tokenizer, device, prune_n=0, prune_m=0, no_reconstruct=False, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):  
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)    

        gpts = {}
        for name in subset:
            if dual_ascent:
                gpts[name] = SparseGPTV3(subset[name], valid=valid_out)
            else:
                gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            gpts[name].fasterprune(ratios[k], prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    gpts[name].validate()
                
                flag, alpha, beta = gpts[name].get_args()
                min_iter = 0
                
                if ratios[k] < args.dual_sp_theld:
                    flag = False

                if flag:
                    gpts[name].dual_ascent4(beta = beta, alpha = alpha, epsilon=args.epsilon, min_iter=min_iter, theld= args.dual_theld, alpha_2=True)
                    if debug:
                        print("Validation after dual ascent:")
                        gpts[name].validate()
                    gpts[name].del_valid()
                else:
                    del gpts[name].H, gpts[name].H_B

            gpts[name].free()
            k+=1

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()

        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda
        torch.cuda.empty_cache()
        # if debug:
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/inps_layer{i}.npy", inps.cpu().numpy())
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/outs_layer{i}.npy", outs.cpu().numpy())
        print(f"layer {i} done")
        # print("Inps max and mean: ", outs.max(), outs.abs().mean())

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_sparsegpt_5(args, model, tokenizer, device, prune_n=0, prune_m=0, no_reconstruct=False, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):  
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device

    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)    

        gpts = {}
        for name in subset:
            if dual_ascent:
                gpts[name] = SparseGPTV3(subset[name], valid=valid_out)
            else:
                gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            gpts[name].fasterprune2(ratios[k], prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)

            if dual_ascent:
                if debug:
                    print("Validation after prune:")
                    gpts[name].validate()
                
                flag, alpha, beta = gpts[name].get_args()
                min_iter = 0
                
                if ratios[k] < args.dual_sp_theld:
                    flag = False

                if flag:
                    gpts[name].dual_ascent5(beta = beta, alpha = alpha, epsilon=args.epsilon, min_iter=min_iter, theld= args.dual_theld, alpha_2=True)
                    if debug:
                        print("Validation after dual ascent:")
                        gpts[name].validate()
                    gpts[name].del_valid()
                else:
                    del gpts[name].H, gpts[name].H_B, gpts[name].ora_W

            gpts[name].free()
            k+=1

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()

        layer_cuda = layer_cuda.cpu()
        layers[i] = layer_cuda
        torch.cuda.empty_cache()
        # if debug:
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/inps_layer{i}.npy", inps.cpu().numpy())
        #     np.save(f"/h3cstore_ns/jcxie/LISA/wanda-main/npys/debug/outs_layer{i}.npy", outs.cpu().numpy())
        print(f"layer {i} done")
        # print("Inps max and mean: ", outs.max(), outs.abs().mean())

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_sparsegpt_pcg(args, model, tokenizer, device, prune_n=0, prune_m=0, no_reconstruct=False, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):
    pass

@torch.no_grad()
def prune_dsnot(args, model, tokenizer, device, prune_n=0, prune_m=0, no_reconstruct=False, exclude=None, ratios=None, dual_ascent=False, cuda_device="cuda"):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    dev = cuda_device
    total_time = 0
    print("loading calibdation data")
    dataloader, _ = get_loaders(args, "c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(args, model, dataloader, device, nsamples=args.nsamples)

    layers = model.model.layers

    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer_cuda = layers[i].to(dev)
        if exclude is not None:
            subset = find_layers_without(layer_cuda, exclude=exclude)
        else:
            subset = find_layers(layer_cuda)

        try:    
            dev = model.hf_device_map[f"model.layers.{i}"]
        except:
            dev = dev
        if attention_mask is None:
            position_ids = position_ids.to(dev)
        else:
            attention_mask, position_ids = attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            if dual_ascent:
                wrapped_layers[name] = WrappedGPTV10(subset[name], valid=valid_out)
            else:
                wrapped_layers[name] = WrappedGPTV10(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        batch_num = int(args.nsamples / args.n_batch)

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()  

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            start_time = time.time()

            DSnoT_metric = subset[name].weight.data * wrapped_layers[name].sum_metric_row.reshape((1, -1))

            if "wanda" in args.prune_method:
                initial_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )
            elif "magnitude" in args.prune_method:
                initial_metric = torch.abs(subset[name].weight.data)
            elif "sparsegpt" in args.prune_method:
                W = subset[name].weight.data.clone()
                if isinstance(subset[name], nn.Conv2d):
                    W = W.flatten(1)
                if isinstance(subset[name], transformers.Conv1D):
                    W = W.t()
                W = W.float()

                H = wrapped_layers[name].H
                # del wrapped_layers[name].H
                dead = torch.diag(H) == 0
                H[dead, dead] = 1
                W[:, dead] = 0

                percdamp = 0.01
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(
                    wrapped_layers[name].columns, device=wrapped_layers[name].dev
                )
                H[diag, diag] += damp
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H

                initial_metric = W**2 / (torch.diag(Hinv).reshape((1, -1))) ** 2

            weight_mask = torch.zeros_like(initial_metric) == 1

            if prune_n != 0:
                raise NotImplementedError(
                    "Structured pruning is not implemented in DSnoT pruning."
                )
            else:
                _, sorted_initial_indice = torch.sort(
                    initial_metric, dim=-1, stable=True
                )

                sparsity_num = int(initial_metric.shape[1] * ratios[k])
                k += 1
                res_sparsity_num = sorted_initial_indice.shape[1] - sparsity_num

                initial_prune_indices, initial_res_indices = torch.split(
                    sorted_initial_indice,  
                    split_size_or_sections=[sparsity_num, res_sparsity_num],
                    dim=1,
                )

                if (
                    name.split(".")[0] == args.skip_layer
                    or name.split(".")[1] == args.skip_sub_layer
                    or args.without_DSnoT
                ):
                    weight_mask.scatter_(1, initial_prune_indices, True)

                else:
                    weight_mask.scatter_(1, initial_prune_indices, True)

                    metric_for_regrowing = DSnoT_metric.clone()
                    wanda_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                        wrapped_layers[name].scaler_row.reshape((1, -1))
                    )

                    metric_for_regrowing.scatter_(1, initial_res_indices, 0)
                    reconstruction_error = torch.sum(
                        metric_for_regrowing, dim=1, keepdim=True
                    )
                    initialize_error_sign = torch.sign(reconstruction_error)

                    if args.pow_of_var_regrowing:
                        metric_for_regrowing /= torch.pow(
                            wrapped_layers[name].var.reshape((1, -1)),
                            args.pow_of_var_regrowing,
                        )

                    _, regrowing_indices_block = torch.sort(
                        metric_for_regrowing, dim=1, stable=True
                    )

                    wanda_metric.scatter_(1, initial_prune_indices, float("inf"))
                    wanda_res_indices, _ = torch.split(
                        torch.sort(wanda_metric, dim=1, stable=True)[1],
                        split_size_or_sections=[res_sparsity_num, sparsity_num],
                        dim=1,
                    )
                    reorder_indice_of_pruning_indice = return_reorder_indice(
                        torch.gather(DSnoT_metric, 1, wanda_res_indices)
                    )
                    pruning_indices_block = torch.gather(
                        wanda_res_indices, 1, reorder_indice_of_pruning_indice
                    )

                    indice_indice_list_for_regrowing = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = regrowing_indices_block.shape[-1] - 1
                    indice_indice_list_for_regrowing[:, 1] = last_one

                    update_num_for_regrowing = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_regrowing[:, 1] = -1

                    indice_indice_list_for_pruning = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = pruning_indices_block.shape[-1] - 1
                    indice_indice_list_for_pruning[:, 1] = last_one

                    update_num_for_pruning = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_pruning[:, 1] = -1

                    update_mask = torch.ones_like(
                        reconstruction_error, dtype=torch.bool
                    )
                    cycle_time = 0
                    while not ( torch.all(update_mask == False) or cycle_time >= args.max_cycle_time ):
                        cycle_time += 1
                        
                        # regrowing
                        indice_of_indice_indice_list_for_regrowing = (
                            (reconstruction_error > 0).int().to(torch.int64)
                        )

                        indice_indice_for_regrowing = torch.gather(
                            indice_indice_list_for_regrowing,
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                        )

                        regrowing_indice = torch.gather(
                            regrowing_indices_block,
                            1,
                            indice_indice_for_regrowing.to(torch.int64),
                        )

                        regrowing_metric = DSnoT_metric.gather(
                            1, regrowing_indice.to(torch.int64)
                        )

                        indice_indice_list_for_regrowing.scatter_(
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                            indice_indice_for_regrowing
                            + update_num_for_regrowing.gather(
                                1, indice_of_indice_indice_list_for_regrowing
                            ),
                        )

                        # pruning
                        indice_of_indice_indice_list_for_pruning = (
                            (reconstruction_error < 0).int().to(torch.int64)
                        )

                        indice_indice_for_pruning = torch.gather(
                            indice_indice_list_for_pruning,
                            1,
                            indice_of_indice_indice_list_for_pruning,
                        )

                        pruning_indice = torch.gather(
                            pruning_indices_block,
                            1,
                            indice_indice_for_pruning.to(torch.int64),
                        )

                        pruning_metric = DSnoT_metric.gather(
                            1, pruning_indice.to(torch.int64)
                        )

                        indice_indice_list_for_pruning.scatter_(
                            1,
                            indice_of_indice_indice_list_for_pruning, 
                            indice_indice_for_pruning
                            + update_num_for_pruning.gather(
                                1, indice_of_indice_indice_list_for_pruning
                            ),
                        )

                        # change mask
                        reconstruction_error_after = (
                            reconstruction_error + pruning_metric - regrowing_metric
                        )

                        if args.without_same_sign == str(True):
                            update_mask = update_mask & (
                                abs(reconstruction_error) > args.update_threshold
                            )
                        else:
                            update_mask = (
                                update_mask
                                & (abs(reconstruction_error) > args.update_threshold)
                                & (
                                    initialize_error_sign
                                    == torch.sign(reconstruction_error_after)
                                )
                            )

                        weight_mask.scatter_(1, pruning_indice, update_mask)
                        weight_mask.scatter_(1, regrowing_indice, ~update_mask)

                        reconstruction_error += torch.where(
                            update_mask,
                            pruning_metric,
                            torch.zeros_like(pruning_metric),
                        )
                        reconstruction_error -= torch.where(
                            update_mask,
                            regrowing_metric,
                            torch.zeros_like(regrowing_metric),
                        )

            
            subset[name].weight.data[weight_mask] = 0

            end_time = time.time()
            total_time += end_time - start_time

        for b in range(batch_num):
            i1 = b * args.n_batch
            i2 = (b + 1) * args.n_batch

            inps_cuda = inps[i1:i2].to(dev)
            outs_cuda = torch.zeros_like(inps_cuda)

            for j in range(args.n_batch):
                if attention_mask is None:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), position_ids=position_ids)[0]
                else:
                    outs_cuda[j] = layer_cuda(inps_cuda[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            outs_cuda = outs_cuda.cpu()
            inps_cuda = inps_cuda.cpu()
            outs[i1:i2] = outs_cuda
            torch.cuda.empty_cache()  


    model.config.use_cache = use_cache
    torch.cuda.empty_cache()