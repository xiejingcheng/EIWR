# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
from tqdm import tqdm
from torch import nn

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
# def get_wikitext2(nsamples, seed, seqlen, tokenizer):
#     # Load train and test datasets
#     traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
#     testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

#     # Encode datasets
#     trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
#     testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

#     # Generate samples from training set
#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))
#     return trainloader, testenc

# # Load and process c4 dataset
# def get_c4(nsamples, seed, seqlen, tokenizer):
#     # Load train and validation datasets
#     traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
#     valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

#     # Generate samples from training set
#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         while True:
#             i = random.randint(0, len(traindata) - 1)
#             trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
#             if trainenc.input_ids.shape[1] > seqlen:
#                 break
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))

#     # Prepare validation dataset
#     valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
#     valenc = valenc.input_ids[:, :(256 * seqlen)]
#     valenc = TokenizerWrapper(valenc)
#     return trainloader, valenc


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    
    traindata = load_dataset(
        "parquet",  # 数据格式是 Parquet
        data_files={"/h3cstore_ns/jcxie/hf_dataset/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet"},
        split="train"
    )

    testdata = load_dataset(
        "parquet",  # 数据格式是 Parquet
        data_files={"/h3cstore_ns/jcxie/hf_dataset/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"},
        split="train"
)

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('/h3cstore_ns/jcxie/hf_dataset/ptb_text_only/ptb_text_only.py', split='train')
    testdata = load_dataset('/h3cstore_ns/jcxie/hf_dataset/ptb_text_only/ptb_text_only.py', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset(
        'json',  # 数据格式是 JSON
        data_files={'train': '/h3cstore_ns/jcxie/hf_dataset/c4/c4-train.00000-of-01024.json.gz'}, 
        split='train'
    )
    # 加载本地验证数据
    valdata = load_dataset(
        'json',  # 数据格式是 JSON
        data_files={'validation': '/h3cstore_ns/jcxie/hf_dataset/c4/c4-validation.00000-of-00008.json.gz'}, 
        split='validation'
    )

    random.seed(seed)
    trainloader = []
    for _ in tqdm(range(nsamples)):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


# Function to select the appropriate loader based on dataset name
def get_loaders(args, name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if "Qwen" in args.model or "phi" in args.model:
        seqlen = 4096

    if seqlen > 4096:
        seqlen = 4096

    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    
def prepare_calibration_input(args, model, dataloader, device, nsamples=128):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    try:
        if "model.embed_tokens" in model.hf_device_map:
            device = model.hf_device_map["model.embed_tokens"]
    except:
        device = torch.device("cuda")

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.rotary_emb = model.model.rotary_emb.to(device)

    dtype = next(iter(model.parameters())).dtype
    if "Qwen" in args.model or "phi" in args.model:
        model_seqlen = 4096
    else:
        model_seqlen = model.seqlen
    if model_seqlen > 4096:
        model_seqlen = 4096
    inps = torch.zeros((nsamples, model_seqlen, model.config.hidden_size), dtype=dtype)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    print("Preparing calibration input...")
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    model.model.embed_tokens = model.model.embed_tokens.to("cpu")
    model.model.rotary_emb = model.model.rotary_emb.to("cpu")
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def prepare_calibration_input_sam(args, model, dataloader, nsamples, seqlen=64):
    model.eval()
    if "vit_b" in args.model_name:
        hidden_size = 768
    elif "vit_l" in args.model_name:
        hidden_size = 1024
    elif "vit_h" in args.model_name:
        hidden_size = 1280
    # print("preparing calibration input")
    # print("nsamples: ", nsamples)
    layers = model.image_encoder.blocks
    inps = torch.zeros((nsamples, seqlen, seqlen, hidden_size), dtype=torch.float32, device=model.device)
    # print("inps size: ", inps.size())
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # print(inp.size())
            # print("input size: ", inp.size())
            # print(cache['i'])
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for i, batch in enumerate(dataloader):
        try:
            input_image, labels = batch['image'], batch['label']
            input_image = input_image.cuda()
            model.image_encoder(input_image)
        except ValueError:
            pass 
        
        if cache['i'] == nsamples-1:
            break
    layers[0] = layers[0].module

    out = torch.zeros_like(inps, dtype=torch.float32, device=model.device)

    return inps, out
