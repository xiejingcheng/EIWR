CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.0  &

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.1  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.2  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.3  &

wait && \
(
    CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
        --prune_method wanda \
        --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
        --sparsity_ratio 0.4  &

    CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
        --prune_method wanda \
        --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
        --sparsity_ratio 0.5  &

    CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
        --prune_method wanda \
        --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
        --sparsity_ratio 0.6  &

    CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
        --prune_method wanda \
        --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
        --sparsity_ratio 0.7  &
    
    wait 
) && \
(
    CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
        --prune_method wanda \
        --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
        --sparsity_ratio 0.8  &

    CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
        --prune_method wanda \
        --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
        --sparsity_ratio 0.9  &

    wait 
)