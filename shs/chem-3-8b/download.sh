CUDA_VISIBLE_DEVICES='0,1' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt \
    --model /h3cstore_ns/jcxie/hf_weights/Llama3-KALE-LM-Chem-1.5-8B \
    --save_ckpt 1 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.5  &


CUDA_VISIBLE_DEVICES='2,3' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual_3 \
    --model /h3cstore_ns/jcxie/hf_weights/Llama3-KALE-LM-Chem-1.5-8B \
    --save_ckpt 1 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.45  &

wait

