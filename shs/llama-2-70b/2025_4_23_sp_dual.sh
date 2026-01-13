/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_eff_v2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 256 \
    --save_ckpt 1 \
    --sparsity_ratio 0.7 &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_eff_v2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 256 \
    --sparsity_ratio 0.8 &

wait

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_eff_v2 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 256 \
    --sparsity_ratio 0.6 &

wait