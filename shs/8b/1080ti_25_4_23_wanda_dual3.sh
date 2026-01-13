/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.6  &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.7  &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --dual_theld 0.03 \
    --sparsity_ratio 0.8  &

wait