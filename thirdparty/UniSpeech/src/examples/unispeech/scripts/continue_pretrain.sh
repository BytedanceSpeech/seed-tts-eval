#:: Copyright (c) Microsoft Corporation.
#:: Licensed under the MIT License.

model_path=MODEL_PATH
train_subset=pretrain_HOUR_16k
valid_subset=valSeqs_1.0_uniform_new_version_16k
WORLD_SIZE=8


update_freq=2

mkdir -p ${model_path}

python train.py --distributed-world-size ${WORLD_SIZE} --distributed-port 0 examples/unispeech/data/LANG --save-dir ${model_path} --fp16 --num-workers 10 --task audio_pretraining --criterion wav2vec --arch wav2vec2 --train-subset ${train_subset} --valid-subset ${valid_subset} --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --normalize --extractor-mode "layer_norm" --encoder-layers 24 --encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --encoder-attention-heads 16 --final-dim 768 --layer-norm-first --conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --latent-vars 320 --latent-groups 2 --latent-temp '(2,0.1,0.999995)' --infonce --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 100000 --lr 0.0002 --warmup-updates 10000 --mask-length 10 --mask-prob 0.65 --mask-selection static --mask-other 0 --encoder-layerdrop 0.05 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 0.1 --loss-weights '[0.1, 0]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 --max-sample-size 250000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --max-tokens 1000000 --max-update 100000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --update-freq ${update_freq} --pretrained-path PRETRAINED_MODEL --no-epoch-checkpoints --transpose
