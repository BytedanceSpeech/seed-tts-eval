#:: Copyright (c) Microsoft Corporation.
#:: Licensed under the MIT License.

model_path=MODEL_PATH
train_subset=pretrain_1350_16k
valid_subset=valid_16k
WORLD_SIZE=NUM_OF_GPUS


update_freq=$((64/$WORLD_SIZE))  #ngpu * update_freq = 64

mkdir -p ${model_path}

python train.py --distributed-world-size ${WORLD_SIZE} --distributed-port 0 examples/unispeech/data/en --save-dir ${model_path} --fp16 --num-workers 10 --task audio_pretraining --criterion unispeech_criterion --arch unispeech --extractor-mode "layer_norm" --encoder-layers 24 --encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --encoder-attention-heads 16 --final-dim 768 --layer-norm-first --conv-bias --logit-temp 0.1 --train-subset ${train_subset} --valid-subset ${valid_subset} --log-keys '["prob_perplexity","code_perplexity","temp"]' --quantize-targets --conv-feature-layers '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2' --latent-vars 320 --latent-groups 2 --latent-temp '(2,0.1,0.999995)' --infonce --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-06 --lr-scheduler polynomial_decay --total-num-update 200000 --lr 0.001 --warmup-updates 25000 --mask-length 10 --mask-prob 0.5 --mask-selection static --mask-other 0 --encoder-layerdrop 0.0 --dropout-input 0.1 --dropout-features 0.1 --feature-grad-mult 1.0 --loss-weights '[0.1, 0]' --conv-pos 128 --conv-pos-groups 16 --num-negatives 100 --cross-sample-negatives 0 --max-sample-size 320000 --min-sample-size 32000 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --max-tokens 1200000 --max-update 250000 --skip-invalid-size-inputs-valid-test --ddp-backend no_c10d --update-freq ${update_freq} --post-process none --labels id --dict-path examples/unispeech/data/en/vocab.json --negatives-from-everywhere --mtlalpha 0.5 --replace-prob 0.5 --transpose --no-epoch-checkpoints --log-format json
