#:: Copyright (c) Microsoft Corporation.
#:: Licensed under the MIT License.

model_path=MODEL_PATH
gen_subset=testSeqs_uniform_new_version_16k
result_path=${model_path}/decode_ctc/${gen_subset}

mkdir -p ${result_path}

export PYTHONENCODING=UTF-8

python examples/speech_recognition/infer.py examples/unispeech/data/LANG --task audio_pretraining --nbest 1 --path ${model_path}/checkpoint_best.pt --gen-subset ${gen_subset} --results-path ${result_path} --w2l-decoder viterbi --word-score -1 --sil-weight 0 --criterion ctc --max-tokens 4000000 --dict-path examples/unispeech/data/LANG/phonesMatches_reduced.json --post-process none --quiet
