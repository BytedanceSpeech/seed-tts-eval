pip install torch_complex

model_path=MODEL_PATH
data_path=DATA_PATH
label_path=LABEL_PATH
train_subset=train_960
valid_subset=valid

distributed_world_size=WORLD_SIZE
update_freq=$((32/$WORLD_SIZE))

max_tokens=1400000
warmup_updates=32000
total_num_update=400000

mkdir -p ${model_path}

python train.py   \
  --ddp-backend no_c10d \
  --distributed-backend 'nccl' \
  --distributed-world-size ${distributed_world_size}   \
  --distributed-port 29671 \
  --nprocs-per-node 8 \
  --find-unused-parameters \
  --fp16   \
  --log-format json   \
  --log-interval 200   \
  --seed 1337    \
  --save-dir ${model_path}   \
  --save-interval-updates 5000   \
  --keep-interval-updates 10  \
  --no-epoch-checkpoints   \
  --num-workers 6   \
  --task hubert_pretraining   \
  --criterion hubert   \
  --arch ils_hubert   \
  --train-subset ${train_subset}   \
  --valid-subset ${valid_subset}   \
  --log-keys '[]'   \
  ${data_path}   \
  --label-dir ${label_path}   \
  --labels '["km"]'   \
  --sample-rate 16000   \
  --max-sample-size 250000   \
  --min-sample-size 32000   \
  --max-tokens ${max_tokens}   \
  --skip-invalid-size-inputs-valid-test   \
  --validate-interval 5   \
  --validate-interval-updates 10000   \
  --pred-masked-weight 1.0   \
  --pred-nomask-weight 0.0   \
  --loss-weights [10,]   \
  --label-rate 50   \
  --mask-prob 0.80   \
  --extractor-mode default   \
  --conv-feature-layers '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2'   \
  --final-dim 256   \
  --encoder-layerdrop 0.05   \
  --dropout-input 0.1   \
  --dropout-features 0.1   \
  --dropout 0.1   \
  --attention-dropout 0.1   \
  --feature-grad-mult 0.1   \
  --activation-dropout 0.0   \
  --optimizer adam   \
  --adam-betas '(0.9,0.98)'   \
  --adam-eps 1e-06   \
  --weight-decay 0.01   \
  --lr-scheduler polynomial_decay   \
  --warmup-updates ${warmup_updates}   \
  --total-num-update ${total_num_update}   \
  --max-update 400000   \
  --lr 0.0005   \
  --clip-norm 10.0 \
  --update-freq ${update_freq} \
  --predict-layers "[4,12]" \
  --relative-position-embedding \
  --num-buckets 320 \
  --max-distance 800 \
  --required-batch-size-multiple 1 \
  --separate-label-embeds 
