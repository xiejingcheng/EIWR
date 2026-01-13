CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.5  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.6  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.7  &

wait

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.8  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.9  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.05 \
    --sparsity_ratio 0.5  &

wait

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.05 \
    --sparsity_ratio 0.6  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.05 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.05 \
    --sparsity_ratio 0.8  &

wait

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.05 \
    --sparsity_ratio 0.9  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.5  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.6  &

wait

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.8  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.9  &

wait

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.5  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.6  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.7  &

wait

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.8  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --sparsity_ratio 0.9  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.05 \
    --sparsity_ratio 0.5  &

wait

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.05 \
    --sparsity_ratio 0.6  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.05 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.05 \
    --sparsity_ratio 0.8  &

wait

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.05 \
    --sparsity_ratio 0.9  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.5  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.6  &

wait

CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.8  &

CUDA_VISIBLE_DEVICES='6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.9  &

wait


