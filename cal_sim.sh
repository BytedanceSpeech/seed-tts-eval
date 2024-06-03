set -x

meta_lst=$1
output_dir=$2
checkpoint_path=$3

wav_wav_text=$output_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

python3 get_wav_res_ref_text.py $meta_lst $output_dir $output_dir/wav_res_ref_text

workdir=$(cd $(dirname $0); pwd)

cd $workdir/thirdparty/UniSpeech/downstreams/speaker_verification/

timestamp=$(date +%s)
thread_dir=/tmp/thread_metas_$timestamp/
mkdir $thread_dir
num_job=$ARNOLD_WORKER_GPU
num=`wc -l $wav_wav_text | awk -F' ' '{print $1}'`
num_per_thread=`expr $num / $num_job + 1`
sudo split -l $num_per_thread --additional-suffix=.lst -d $wav_wav_text $thread_dir/thread-
out_dir=/tmp/thread_metas_$timestamp/results/
mkdir $out_dir

num_job_minus_1=`expr $num_job - 1`
if [ ${num_job_minus_1} -ge 0 ];then
    for rank in $(seq 0 $((num_job - 1))); do
        python3 verification_pair_list_v2.py $thread_dir/thread-0$rank.lst \
            --model_name wavlm_large \
            --checkpoint $checkpoint_path \
            --scores $out_dir/thread-0$rank.sim.out \
            --wav1_start_sr 0 \
            --wav2_start_sr 0 \
            --wav1_end_sr -1 \
            --wav2_end_sr -1 \
            --device cuda:$rank &
    done
fi
wait

rm $wav_wav_text
rm -f $out_dir/merge.out

cat $out_dir/thread-0*.sim.out | grep -v "avg score" >>  $out_dir/merge.out
python3 average.py $out_dir/merge.out $score_file
