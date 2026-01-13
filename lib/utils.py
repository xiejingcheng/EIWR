import torch
from torch import nn
import numpy as np
# from .matmul_had import *

# def RHT_H(H, SU):
#     return matmul_hadUt(matmul_hadUt(H * SU).T * SU)


# def RHT_W(W, SU, SV):
#     return matmul_hadUt(matmul_hadUt(W.T * SV).T * SU)

# def incoherence_preprocess(H, W, args):
#     dtype_ = torch.float64 if args.use_fp64 else torch.float32
#     device = H.device
#     (m, n) = W.shape

#     def _dump(Hr, Lhr, msg=''):
#         torch.save(Hr, f"{args.save_pfx}/Hr_debug_fft.pt")
#         torch.save(Lhr, f"{args.save_pfx}/Lhr_debug_fft.pt")
#         raise Exception(msg)

#     # diagonally rescale W,H to minimize proxy loss
#     scaleWH = None
#     Wr = W
#     Hr = H
#     if args.rescale_WH: #False
#         Hr = H / H.abs().max()
#         diagH = torch.diag(Hr)
#         diagW2 = torch.diag(W.T @ W)
#         diagH = torch.clamp(diagH, min=1e-8)
#         diagW2 = torch.clamp(diagW2, min=1e-8)
#         scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
#         scaleWH = scaleWH.clamp(min=1e-8)
#         Wr = Wr * scaleWH[None, :]
#         Hr = Hr / scaleWH[None, :]
#         Hr = Hr / scaleWH[:, None]
#         scaleWH = scaleWH.cpu()

#     # randomized hadamard transformation on H, W
#     if args.incoh_mode == "had":
#         SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
#         SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)
#         Hr = RHT_H(Hr, SU)
#         Wr = RHT_W(Wr, SU, SV)
#     # randomized kronecker product on H, W
#     # elif args.incoh_mode == "kron":
#     #     SU = utils.rand_ortho_butterfly_noblock(n).to(dtype_).to(device)
#     #     SV = utils.rand_ortho_butterfly_noblock(m).to(dtype_).to(device)
#     #     Hr = SU @ Hr @ SU.T
#     #     Wr = SV @ Wr @ SU.T
#     else:
#         raise NotImplementedError
#     SV = SV.cpu()
#     SU = SU.cpu()

#     # Lhr = torch.linalg.cholesky(Hr)
#     # if not torch.all(torch.isfinite(Lhr)):
#     #     return None

#     Wr = Wr.to(device)

#     return Hr, Wr, SU, SV, scaleWH

# def incoherence_process(hatWr, SU, SV, scaleWH, args):
#     device = hatWr.device
#     # reverse hadamard transformation
#     if args.incoh_mode == 'had':
#         hatWr = (matmul_hadU(
#             (matmul_hadU(hatWr) * SU.to(device)).T) * SV.to(device)).T
#     # reverse kronecker product
#     elif args.incoh_mode == 'kron':
#         hatWr = SV.T.to(device) @ hatWr @ SU.to(device)
#     else:
#         raise NotImplementedError

#     # reverse rescale W,H
#     if args.rescale_WH:
#         hatWr /= scaleWH[None, :].to(device)

#     assert torch.isfinite(hatWr).all()
#     return hatWr

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def replace_layers(module, gpts, layer_dict, RHT_class):
    """
    Replace the layers specified in the layer_dict with the given RHT_class.

    Args:
        module (nn.Module): PyTorch module.
        layer_dict (dict): Dictionary of layers to replace, as returned by find_layers.
        RHT_class (class): The replacement class for the layers.

    Returns:
        None: The module is modified in place.
    """
    for layer_name, layer_instance in layer_dict.items():
        # 将层名称分解为父模块和子模块
        parent_name, child_name = layer_name.rsplit('.', 1) if '.' in layer_name else ('', layer_name)
        parent_module = module if parent_name == '' else dict(module.named_modules())[parent_name]
        # 用 RHT_class 替换原始层
        RHT_Layer = RHT_class(layer_instance)
        setattr(parent_module, child_name, RHT_Layer)
        layer_dict[layer_name] = RHT_Layer
        gpts[layer_name].layer = RHT_Layer

def find_silu_layers(module, layers=[nn.Linear], name='', silu_name='gate_proj'):
    if type(module) in layers:
        if silu_name in name:
            return {name: module}
        else:
            return {}
    res = {}
    for name1, child in module.named_children():
        res.update(find_silu_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def find_targer_layers(module, layers=[nn.Linear], name='', targer_name='gate_proj'):
    if type(module) in layers:
        for tar_name in targer_name:
            if tar_name in name:
                return {name: module}
        return {}
    res = {}
    for name1, child in module.named_children():
        res.update(find_targer_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1, targer_name=targer_name
        ))
    return res

def find_layers_without(module, layers=[nn.Linear], name='', exclude=None):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if exclude is None:
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res
    else:
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            if name1 != exclude:
                res.update(find_layers_without(
                    child, layers=layers, name=name + '.' + name1 if name != '' else name1, exclude=exclude
                ))
        return res
    
def ww_sparsity(args, model, device=torch.device("cuda:0"), s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0):
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers
    
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())

    layer_num_in_block = int(len(prunables) / len(blocks))

    metrics = np.load(f"{args.ww_metric_cache}/{args.ww_metric}.npy")
    
    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
    
    print("metric values:", metrics)
            
    scores = torch.tensor(metrics)
    prunables = torch.tensor(prunables)

    # linear mapping
    max = torch.max(scores)
    min = torch.min(scores)
    
    layerwise_pruning_ratios = (((scores - min) / (max - min)) * (s2 - s1) + s1)
    scaler = torch.sum(prunables) * args.sparsity_ratio / (torch.sum(prunables * layerwise_pruning_ratios))  
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    layerwise_pruning_ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    
    print(layerwise_pruning_ratios)
    return layerwise_pruning_ratios

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def check_sparsity_sam(model):
    layers = model.image_encoder.blocks
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    return float(count)/total_params 


def check_outlier_mean(mask,threshold):
    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()



    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity