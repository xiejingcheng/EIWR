CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/Qwen2.5-7b \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/Qwen2.5-7b \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/Qwen2.5-3b \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/Qwen2.5-3b \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

wait

CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/DSnoT-main/main.py \
    --prune_method DSnoT \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --initial_method wanda \
    --save_model /h3cstore_ns/jcxie/LISA/nips2024/log2 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/DSnoT-main/main.py \
    --prune_method DSnoT \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --initial_method sparsegpt \
    --save_model /h3cstore_ns/jcxie/LISA/nips2024/log2 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

wait

CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/DSnoT-main/main.py \
    --prune_method DSnoT \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --initial_method wanda \
    --save_model /h3cstore_ns/jcxie/LISA/nips2024/log2 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/DSnoT-main/main.py \
    --prune_method DSnoT \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --initial_method sparsegpt \
    --save_model /h3cstore_ns/jcxie/LISA/nips2024/log2 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

wait