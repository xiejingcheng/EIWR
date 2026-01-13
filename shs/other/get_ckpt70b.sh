/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 128 \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 128 \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &

wait

