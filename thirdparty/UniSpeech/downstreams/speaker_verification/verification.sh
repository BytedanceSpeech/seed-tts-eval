wav_wav_text=$1
score_file=$2

python3 verification_pair_list_v2.py $wav_wav_text --model_name wavlm_large --checkpoint $PWD/wavlm_large_finetune.pth --scores $score_file --wav1_start_sr 0 --wav2_start_sr 0 --wav1_end_sr -1 --wav2_end_sr -1
