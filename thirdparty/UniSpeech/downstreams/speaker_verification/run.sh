export https_proxy=http://bj-rd-proxy.byted.org:3128 http_proxy=http://bj-rd-proxy.byted.org:3128 no_proxy=code.byted.org

thread_dir=/mnt/bn/jcong5/pretrain_data/process_rp_prompt/
# num_job=4
# for rank in $(seq 1 $((num_job - 1))); do
#     echo $rank
#     python3 extract_embedding.py \
#         --infile $thread_dir/thread-0$rank.lst \
#         --device cuda:$rank &
# done

num_job=8
for rank in $(seq 0 $((num_job - 1))); do
    echo $rank
    part=`expr $rank + 4`
    padded=$(printf "%02d\n" $part)
    echo thread-$padded.lst
    python3 extract_embedding.py \
        --infile $thread_dir/thread-$padded.lst \
        --device cuda:$rank &
done