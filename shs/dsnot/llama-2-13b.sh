CUDA_VISIBLE_DEVICES='0,1,2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/DSnoT-main/main.py \
    --prune_method DSnoT \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-13b-hf \
    --initial_method wanda \
    --sparsity_ratio 0.5  &

CUDA_VISIBLE_DEVICES='4,5,6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/DSnoT-main/main.py \
    --prune_method DSnoT \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-13b-hf \
    --initial_method wanda \
    --sparsity_ratio 0.6  &

wait

CUDA_VISIBLE_DEVICES='0,1,2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/DSnoT-main/main.py \
    --prune_method DSnoT \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-13b-hf \
    --initial_method wanda \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='4,5,6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/DSnoT-main/main.py \
    --prune_method DSnoT \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-13b-hf \
    --initial_method wanda \
    --sparsity_ratio 0.8  &

wait

CUDA_VISIBLE_DEVICES='0,1,2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/DSnoT-main/main.py \
    --prune_method DSnoT \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-13b-hf \
    --initial_method sparsegpt \
    --sparsity_ratio 0.5  &

CUDA_VISIBLE_DEVICES='4,5,6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/DSnoT-main/main.py \
    --prune_method DSnoT \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-13b-hf \
    --initial_method sparsegpt \
    --sparsity_ratio 0.6  &

wait

CUDA_VISIBLE_DEVICES='0,1,2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/DSnoT-main/main.py \
    --prune_method DSnoT \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-13b-hf \
    --initial_method sparsegpt \
    --sparsity_ratio 0.7  &

CUDA_VISIBLE_DEVICES='4,5,6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/DSnoT-main/main.py \
    --prune_method DSnoT \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-13b-hf \
    --initial_method sparsegpt \
    --sparsity_ratio 0.8  &

wait