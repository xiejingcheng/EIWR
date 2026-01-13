import argparse
import os 
from datetime import datetime
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_sparsegpt, prune_magnitude
from lib.prune import prune_sparsegpt_ww, prune_wanda_ww
from lib.eval import eval_ppl, eval_zero_shot
from lib.esd_utils import get_esd_metrics

from lib.sam_prune import prune_wanda_sam, prune_sparsegpt_sam

from lib.utils import check_sparsity, check_sparsity_sam

from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map

from segment_anything import sam_model_registry

from segment_anything.utils.dataset import dataset_dis, dataset_dis_val, dataset_duts, dataset_duts_te
from segment_anything.utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from segment_anything.utils import misc
from segment_anything.utils.loss import norm_attn, pca_fit_transform, sig_ce_loss, dice_loss, mask_iou, sig_mae_score, f1_score


print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

debug = False 

def evaluate(args, net, valid_dataloaders, epoch, visualize, valid_datasets=None):

    txt_path = args.save + '/' + args.prune_method + args.model_name + '_sam.txt'
    net.eval()
    results = []
    print("Validating...")
    val_loss = 0.0
    tar_loss = 0.0
    val_cnt = 0.0

    tmp_f1 = []
    tmp_mae = []
    tmp_time = []
    iters = 0
    for k in range(len(valid_dataloaders)):
        valid_dataloader = valid_dataloaders[k]
        val_num = len(valid_dataloader)
        Max_F = np.zeros((val_num))
        Adp_F = np.zeros((val_num))
        E_measure = np.zeros((val_num))
        S_measure = np.zeros((val_num))
        Wgt_F = np.zeros((val_num))
        MAE = np.zeros((val_num))
        HCE = np.zeros((val_num))

        maes = 0
        f1s = 0
        ious = 0

        for i , data in enumerate(valid_dataloader):
            input_image, labels = data['image'], data['label']
            input_image = input_image.cuda()           
            labels = labels.cuda()
            labels_box = misc.masks_to_boxes(labels[:,0,:,:] * 255)
            with torch.no_grad():
                image_embedding = net.image_encoder(input_image)
                sparse_embeddings, dense_embeddings = net.prompt_encoder(
                                                        points=None,
                                                        boxes=labels_box,
                                                        masks=None,
                                                        )
                
                low_res_masks, iou_predictions = net.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=net.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        )

                upscaled_masks = net.postprocess_masks(low_res_masks, args.input_size, args.input_size).cuda()
                mae_score = sig_mae_score(upscaled_masks, labels)
                f1 = f1_score(upscaled_masks, labels)
                iou = mask_iou(upscaled_masks, labels)
                upscaled_masks = torch.sigmoid(upscaled_masks)
                # print(upscaled_masks.max(), upscaled_masks.min())

                

            maes += mae_score.item()
            f1s += f1.item()
            ious += iou.item()

            iters += 1          
            if iters % 100 == 0:
                print(iters)          

            del input_image, labels, image_embedding, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, upscaled_masks
    
            if debug and i > 10:
                break
        print(f'validating on the dataset {k}...')
        # logger.info(f'validating on the dataset {k}...')

        
        with open(txt_path, 'a') as file:
            now = datetime.now()
            file.write('\n')
            formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
            file.write(formatted_now + '\n')
            # name = valid_datasets[k]['name']
            # file.write(f'validating on the dataset {name}...\n')
            file.write('maes: %f, ious: %f, f1s: %f\n'%(maes/val_num, ious/val_num, f1s/val_num))
            print('maes: %f, ious: %f, f1s: %f\n'%(maes/val_num, ious/val_num, f1s/val_num))
            print(args, file=file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/h3cstore_ns/jcxie/SAM/SVD_SAM/pretrain/sam_vit_b_01ec64.pth')
    parser.add_argument('--model_name', type=str, default='vit_b')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=417, help='Number of calibration samples.')
    parser.add_argument('--n_batch', type=int, default=32, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument('--n_flod', type=int, default=1, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, default="sparsegpt_dual_sam", choices=["magnitude", "wanda", "wanda_v2", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search", "sparsegpt_no_reconstruct",
                        "wanda_reconstruct", "wanda_reconstruct_v2", "sparsegpt_exclude", "sparsgpt_silu", "sparsgpt_plus", "sparsegpt_mini", 
                        "wanda_outlier", "wanda_outlier_res", "sparsegpt_ww", "wanda_ww", "wanda_ww_res", "sparsegpt_silu_ww",
                        "sparsegpt_rht", "sparsegpt_wanda", "sparsegpt_uni", "sparsegpt_dual", "wanda_dual", "sparsegpt_dual_batch", "sparsegpt_dual_batch_plus",
                        "sparsegpt_dual_ww", "wanda_dual_sam", "sparsegpt_sam", "wanda_sam", "sparsegpt_dual_sam"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default='/h3cstore_ns/jcxie/LISA/nips2024/log', help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--exclude', type=str, default='gate_proj', help='Layers to exclude from pruning.')

    parser.add_argument("--ww_metric", default="alpha_peak", type=str, help="the WW-based metric to ues.")
    parser.add_argument("--ww_metric_cache", default="/h3cstore_ns/jcxie/LISA/wanda-main/data/llama2-7b-hf")

    parser.add_argument("--wanda_scale", default=0.01, type=float, help="scale for the WANDA pruning.")

    parser.add_argument("--epsilon", default=0.3, type=float, help="for pruning ratio allocation.")
    parser.add_argument("--mapping_type", default="block_wise", type=str, help="mapping type for pruning ratios allocation.")
    parser.add_argument("--dual_theld", default=0.07, type=float, help="mapping layer for pruning ratios allocation.")

    parser.add_argument('--input_size', default=[1024,1024], type=list)

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

    model = sam_model_registry[args.model_name](args.model).cuda()

    valid_datasets = [dataset_dis_val]

    input_size = [1024,1024]
    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                            my_transforms = [
                                                                        Resize(input_size)
                                                                    ],
                                                            batch_size=1,
                                                            training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    if args.prune_method == "wanda_sam":
        prune_wanda_sam(args, model, valid_dataloaders, prune_n, prune_m)
    elif args.prune_method == "wanda_dual_sam":
        prune_wanda_sam(args, model, valid_dataloaders, prune_n, prune_m, dual_ascent=True)
    elif args.prune_method == "sparsegpt_sam":
        prune_sparsegpt_sam(args, model, valid_dataloaders, prune_n, prune_m)
    elif args.prune_method == "sparsegpt_dual_sam":
        prune_sparsegpt_sam(args, model, valid_dataloaders, prune_n, prune_m, dual_ascent=True)

    check_sparsity_sam(model)

    evaluate(args, model, valid_dataloaders, 0, visualize=False, valid_datasets=valid_datasets)

if __name__ == "__main__":
    main()

    