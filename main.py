import argparse
import os 
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_sparsegpt, prune_magnitude, prune_wanda_2, prune_sparsegpt_2
from lib.prune import prune_sparsegpt_ww, prune_wanda_ww, prune_sparsegpt_eff, prune_sparsegpt_3, prune_wanda_3
from lib.prune import prune_sparsegpt_eff_v2, prune_wanda_eff, prune_wanda_eff_v2, prune_sparsegpt_4, prune_sparsegpt_5
from lib.prune import prune_wanda_5, prune_dsnot
from lib.eval import eval_ppl, eval_zero_shot
from lib.esd_utils import get_esd_metrics

from lib.utils import check_sparsity

from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def get_llm_cpu(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True,
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/h3cstore_ns/jcxie/hf_weights/llama-3.2-1b-instruct')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=256, help='Number of calibration samples.')
    parser.add_argument("--dual_theld", default=0.03, type=float, help="mapping layer for pruning ratios allocation.")
    parser.add_argument('--n_batch', type=int, default=32, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--epsilon", default=1e-1, type=float, help="for pruning ratio allocation.")
    parser.add_argument('--n_flod', type=int, default=1, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, default="sparsegpt", choices=["magnitude", "wanda", "sparsegpt", 
                        "sparsegpt_dual", "wanda_dual", "sparsegpt_ww", "wanda_ww", "sparsegpt_dual_ww", 
                        "sparsegpt_dual_ww", "wanda_dual_2", "sparsegpt_dual_2", "sparsegpt_dual_eff", 
                        "sparsegpt_dual_eff_v2", "sparsegpt_dual_3", "wanda_dual_3","magnitude_dual", "wanda_dual_5",
                        "wanda_dual_eff", "wanda_dual_eff_v2", "sparsegpt_dual_4", "sparsegpt_dual_5",
                        "wanda_dsnot", "sparsegpt_dsnot", "magnitude_dsnot",])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default='/h3cstore_ns/jcxie/LISA/nips2024/log2', help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--exclude', type=str, default='gate_proj', help='Layers to exclude from pruning.')

    parser.add_argument("--ww_metric", default="alpha_peak", type=str, help="the WW-based metric to ues.")
    parser.add_argument("--ww_metric_cache", default="/h3cstore_ns/jcxie/LISA/wanda-main/data/llama2-7b-hf")

    parser.add_argument("--wanda_scale", default=0.01, type=float, help="scale for the WANDA pruning.")

    parser.add_argument("--ww_epsilon", default=2e-2, type=float, help="for pruning ratio allocation.")
    
    parser.add_argument("--mapping_type", default="block_wise", type=str, help="mapping type for pruning ratios allocation.")
    
    parser.add_argument("--dual_sp_theld", default=0.3, type=float, help="mapping layer for pruning ratios allocation.")

    parser.add_argument(
        '--Hyper_m', 
        type=float,
        default=3, )
    parser.add_argument(
        "--Lamda",
        default=0.20,
        type=float,
        help="Lamda",
    )

    parser.add_argument("--eval_zero_shot", default=0, type=int, help="whether to eval zero shot tasks")

    parser.add_argument("--save_ckpt", default=0, type=int, help="whether to use accelerate to load the model")

    parser.add_argument('--max_cycle_time', type=int, default=50, help='Max cycle time.')
    parser.add_argument('--without_DSnoT', action="store_true", help="without DSnoT")
    parser.add_argument('--update_threshold', type=float, default=0.1, help='update threshold.')
    parser.add_argument('--pow_of_var_regrowing', type=float, default=1, help='The power of variance.')
    parser.add_argument('--pow_of_var_pruning', type=float, default=1, help='The power of variance.')
    parser.add_argument("--skip_layer", type=str, default="mlp", choices=["no_skip", "mlp", "self_attn"])
    parser.add_argument("--skip_sub_layer", type=str, default="no_skip", choices=["no_skip", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "fc1", "fc2", "out_proj"])
    parser.add_argument('--without_same_sign', type=str, default="True", choices=["True", "False"], help="without same sign")
    parser.add_argument('--eval_ppl', type=int, default=1, help="whether to eval ppl on wikitext2, c4 and ptb")
    args = parser.parse_args()

#----------------------------------------------------------------------------------------------------
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    if "ww" in args.prune_method and not os.path.exists(f"{args.ww_metric_cache}/{args.ww_metric}.npy"):
        metric_values = get_esd_metrics(args.model, args.ww_metric, args.cache_dir)
        np.save(f"{args.ww_metric_cache}/{args.ww_metric}.npy", metric_values)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm_cpu(args.model, args.cache_dir)
    model.eval()
    for n, p in model.named_parameters():
        print(n, p.size())
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    try:
        device = torch.device("cuda:0")
        if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
            device = model.hf_device_map["lm_head"]
        device = model.hf_device_map["lm_head"]
    except:
        device = torch.device("cpu")
    print("use device ", device)
    import time
    start = time.time()
    if args.sparsity_ratio == 0.0:
        print("sparsity ratio is 0, no pruning") 
    else:
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude_dual":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt_dual":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "wanda_dual":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "sparsegpt_ww":
            prune_sparsegpt_ww(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wanda_ww":
            prune_wanda_ww(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt_dual_ww":
            prune_sparsegpt_ww(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "wanda_dual_ww":
            prune_wanda_ww(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "wanda_dual_2":
            prune_wanda_2(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "sparsegpt_dual_2":
            prune_sparsegpt_2(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "sparsegpt_dual_eff":
            prune_sparsegpt_eff(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "sparsegpt_dual_eff_v2":
            prune_sparsegpt_eff_v2(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "sparsegpt_dual_3":
            prune_sparsegpt_3(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "wanda_dual_3":
            prune_wanda_3(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "wanda_dual_eff":
            prune_wanda_eff(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "wanda_dual_eff_v2":
            prune_wanda_eff_v2(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "sparsegpt_dual_4":
            prune_sparsegpt_4(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "sparsegpt_dual_5":
            prune_sparsegpt_5(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "wanda_dual_5":
            prune_wanda_5(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m, dual_ascent=True)
        elif args.prune_method == "wanda_dsnot":
            prune_dsnot(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt_dsnot":
            prune_dsnot(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude_dsnot":
            prune_dsnot(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        


    print("pruning time: ", time.time()-start)
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_pretrained(args.model)

    if '7b' in args.model and 'Qwen' in args.model:
        if torch.cuda.device_count()==2:
            max_memory = {
                0: "15GiB",
                1: "24GiB",  

                "cpu": "0GiB",  
            }
        else:
            max_memory = None

    elif '3b' in args.model:
        if torch.cuda.device_count()==1:
            max_memory = {
                0: "30GiB",
                "cpu": "0GiB",  
            }
        else:
            max_memory = None

    # elif '13b' in args.model:
    #     if torch.cuda.device_count()==2:
    #         max_memory = {
    #             0: "58GiB",
    #             1: "58GiB",  
    #             "cpu": "0GiB",  
    #         }
    #     else:
    #         max_memory = None

    
    elif '70b' in args.model:
        max_memory = {
            0: "38GiB",  
            1: "38GiB",
            2: "38GiB",
            3: "38GiB",
            4: "38GiB",
            5: "38GiB",
            6: "38GiB",
            7: "38GiB",
            "cpu": "0GiB",  
        }
    elif '32b' in args.model:
        max_memory = {
            0: "38GiB",  
            1: "38GiB",
            2: "38GiB",
            3: "38GiB",
            "cpu": "0GiB",  
        }
    else:
        max_memory = None


    device_map = infer_auto_device_map(empty_model, max_memory=max_memory, no_split_module_classes=empty_model._no_split_modules)
    
    model = dispatch_model(model, device_map=device_map)

    print(model.hf_device_map)

    try:
        device = torch.device("cuda:0")
        if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
            device = model.hf_device_map["lm_head"]
        device = model.hf_device_map["lm_head"]
    except:
        device = torch.device("cuda")
    print("use device ", device)
    
    ################################################################
    print("*"*30)
    try:
        with torch.no_grad():
            sparsity_ratio = check_sparsity(model)
    except Exception as e:
        print(f"check sparsity failed: {e}")
        sparsity_ratio = args.sparsity_ratio
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    model_name = args.model.split("/")[-1]
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}_{model_name}.txt")

    if args.save_ckpt:
        save_path = os.path.join(args.save, f"{args.prune_method}_{model_name}_{args.sparsity_ratio:.2f}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    ################################################################
    if "Qwen" in args.model or "phi" in args.model:
        max_seqlen = 4096
    else:
        max_seqlen = 4096

    if args.eval_ppl:
        ppls = []
        for dataset in ["wikitext2","c4"]:
            ppl_test = eval_ppl(args, model, tokenizer, device, max_seqlen=max_seqlen, dataset=dataset)
            ppls.append(ppl_test)
        print(f"wikitext perplexity {ppl_test}")
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}\t{args.nsamples}")
        print(args)
        with open(save_filepath, "a") as f:
            now = datetime.now() 
            formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
            print(f"timestamp: {formatted_now}", file=f, flush=True)
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            for dataset, ppl_test in zip(["wikitext2", "c4", "ptb"], ppls):
                print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)
                print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}\t{args.nsamples}\t{dataset}", file=f, flush=True)
            print(args, file=f, flush=True)
        print(f"done !\n")
        

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        # task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        task_list = ["rte"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        with open(save_filepath, "a") as f:
            print("********************************")
            print("zero_shot evaluation results", file=f, flush=True)
            print(results, file=f, flush=True)

    # if args.save_model:
    #     model.save_pretrained(args.save_model)
    #     tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()