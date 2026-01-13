CUDA_VISIBLE_DEVICES='0,1,4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude_dual \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-13b-v1.5 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.6  &

CUDA_VISIBLE_DEVICES='2,3,6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude_dual \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-13b-v1.5 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.7  &

wait

CUDA_VISIBLE_DEVICES='0,1,4,5' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude_dual \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-13b-v1.5 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.8  &

CUDA_VISIBLE_DEVICES='2,3,6,7' /h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method magnitude_dual \
    --model /h3cstore_ns/jcxie/hf_weights/vicuna-13b-v1.5 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.9  &

wait