model_path=MODEL_PATH
gen_subset=test_clean
result_path=${model_path}/decode_ctc/${gen_subset}

mkdir -p ${result_path}
export PYTHONENCODING=UTF-8

python examples/speech_recognition/infer.py DATA_PATH --task audio_pretraining --nbest 1 --path ${model_path}/checkpoint_best.pt --gen-subset ${gen_subset} --results-path ${result_path} --w2l-decoder viterbi --word-score -1 --sil-weight 0 --criterion ctc --max-tokens 1100000 --dict-path DICT_PATH --post-process letter --quiet 

