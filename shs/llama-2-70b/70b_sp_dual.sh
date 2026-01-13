/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 258 \
    --sparsity_ratio 0.7 &&

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 258 \
    --sparsity_ratio 0.8 &&

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 258 \
    --sparsity_ratio 0.7 &&

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 258 \
    --sparsity_ratio 0.8 &&

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 258 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.7 &&

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 258 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.8 &&

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 258 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.7 &&

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 258 \
    --dual_theld 0.05 \
    --sparsity_ratio 0.8 &&

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 258 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.7 &&

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method sparsegpt_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 258 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.8 &&

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 258 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.7 &&

/h3cstore_ns/jcxie/condaenv/bin/python /h3cstore_ns/jcxie/LISA/nips2024/main.py \
    --prune_method wanda_dual \
    --model /h3cstore_ns/jcxie/hf_weights/llama-2-70b-hf \
    --nsamples 258 \
    --dual_theld 0.03 \
    --sparsity_ratio 0.8 &&