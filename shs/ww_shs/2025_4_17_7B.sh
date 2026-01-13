CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_ww \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --ww_metric_cache /h3cstore_ns/jcxie/LISA/wanda-main/data/llama2-7b-hf \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_ww \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --ww_metric_cache /h3cstore_ns/jcxie/LISA/wanda-main/data/llama2-7b-hf \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_ww \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --ww_metric_cache /h3cstore_ns/jcxie/LISA/wanda-main/data/llama2-7b-hf \
    --sparsity_ratio 0.8  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_ww \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --ww_metric_cache /h3cstore_ns/jcxie/LISA/wanda-main/data/llama2-7b-hf \
    --sparsity_ratio 0.8  &

wait



