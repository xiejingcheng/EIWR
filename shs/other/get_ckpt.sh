CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --nsamples 128 \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --nsamples 128 \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --nsamples 128 \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --nsamples 128 \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

wait && \
(
    CUDA_VISIBLE_DEVICES='0,1,2' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
        --prune_method sparsegpt \
        --model /h3cstore_ns/jcxie/hf_weights/llama-2-13b-hf \
        --nsamples 128 \
        --save_ckpt 1 \
        --sparsity_ratio 0.7  &

    CUDA_VISIBLE_DEVICES='3,4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
        --prune_method wanda \
        --model /h3cstore_ns/jcxie/hf_weights/llama-2-13b-hf \
        --nsamples 128 \
        --save_ckpt 1 \
        --sparsity_ratio 0.7  &

    
    wait 
)