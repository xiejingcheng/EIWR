CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/llama-3-8b-hf \
    --save_ckpt 1 \
    --sparsity_ratio 0.7  &
wait