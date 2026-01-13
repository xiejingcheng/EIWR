CUDA_VISIBLE_DEVICES='0,1,2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_5 \
    --model /h3cstore_ns/jcxie/hf_weights/Qwen2.5-14b \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='4,5,6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_5 \
    --model /h3cstore_ns/jcxie/hf_weights/Qwen2.5-14b \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

wait

CUDA_VISIBLE_DEVICES='0,1,2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/Qwen2.5-14b \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='4,5,6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/Qwen2.5-14b \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

wait

