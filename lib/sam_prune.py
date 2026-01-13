import time 
import heapq 
import torch 
import torch.nn as nn 
import transformers
import numpy as np
from tqdm import tqdm
import logging

from typing import List, Optional, Tuple, Union
from torch import nn

# from .sam_sparsegpt import SparseGPT, SparseGPT_NoReconstruct, SparseGPTV3
from .sam_layerwrapper import WrappedGPT, WrappedGPTV3, SparseGPTV3, SparseGPT
from .data import get_loaders, prepare_calibration_input_sam

from .ablate import AblateGPT 

from .matmul_had import *
from .utils import *

debug = False


valid_out = True


@torch.no_grad()
def prune_wanda_sam(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0, dual_ascent=False, ratios=None):
    dev = device

    inps, outs = prepare_calibration_input_sam(args, model, dataloader[0], args.nsamples)

    layers = model.image_encoder.blocks
    layer_num = len(find_layers(layers))

    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        wrapped_layers = {}
        for name, module in subset.items():
            print(name)
            if dual_ascent:
                wrapped_layers[name] = WrappedGPTV3(subset[name], ratios[k])
            else:
                wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad(): 
                outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()
        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False

            sort_res = torch.sort(W_metric, dim=-1, stable=True)


                # unstructured pruning
            indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
            k+=1
            W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

            if dual_ascent:
                flag, alpha, beta = wrapped_layers[name].get_args()
                min_iter = 0
                if flag:
                    wrapped_layers[name].dual_ascent2(beta = beta, alpha = alpha, min_iter=min_iter, theld= args.dual_theld)
                    wrapped_layers[name].del_valid()
                else:
                    del wrapped_layers[name].H, wrapped_layers[name].H_B


        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0))[0]
        inps, outs = outs, inps

@torch.no_grad()
def prune_sparsegpt_sam(args, model, dataloader, device=torch.device("cuda:0"), prune_n=0, prune_m=0, dual_ascent=False, ratios=None):
    dev = device

    inps, outs = prepare_calibration_input_sam(args, model, dataloader[0], args.nsamples)

    layers = model.image_encoder.blocks
    layer_num = len(find_layers(layers))

    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    k=0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

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

        for j in range(args.nsamples):
            with torch.no_grad(): 
                outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()
        for name in subset:
            print(i, name)
            gpts[name].validate()
            gpts[name].fasterprune(ratios[k], prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)

            if dual_ascent:
                flag, alpha, beta = gpts[name].get_args()
                min_iter = 0
                print("after pruning")
                gpts[name].validate()
                if flag:
                    gpts[name].dual_ascent2(beta = beta, alpha = alpha, min_iter=min_iter, theld= args.dual_theld)
                    print("after dual ascent")
                    gpts[name].validate()
                    gpts[name].del_valid()
                else:
                    del gpts[name].H, gpts[name].H_B
                

            gpts[name].free()
            k+=1

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer
        inps, outs = outs, inps

        
    

    
        


