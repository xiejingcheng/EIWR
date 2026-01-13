CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.6  &

wait

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.7  &

wait

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.8  &

wait

CUDA_VISIBLE_DEVICES='4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.9  &

wait