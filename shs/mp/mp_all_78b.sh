/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.6 &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.7 &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-7b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.8 &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.6 &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.7 &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.8 &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.6 &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.7 &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-7b-v1.5 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.8 &

wait