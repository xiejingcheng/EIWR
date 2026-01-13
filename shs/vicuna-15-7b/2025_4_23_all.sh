CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.6  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.8  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.1  &

wait

CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.5  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.4  &

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.3  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.2  &

wait

CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.6  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.8  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.1  &

wait

CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.5  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.4  &

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.3  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.2  &

wait



