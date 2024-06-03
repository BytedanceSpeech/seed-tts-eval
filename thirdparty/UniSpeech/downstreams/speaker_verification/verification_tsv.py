'''
python verification_tsv.py $tsv1 $tsv2 --model_name wavlm_large --checkpoint wavlm_large_finetune.pth --scores $score_file --wav1_start_sr 0 --wav2_start_sr 0 --wav1_end_sr -1 --wav2_end_sr -1
'''
import tqdm
import argparse
from verification import verification

parser = argparse.ArgumentParser()
parser.add_argument('tsv1')
parser.add_argument('tsv2')
parser.add_argument('--model_name')
parser.add_argument('--checkpoint')
parser.add_argument('--scores')
parser.add_argument('--wav1_start_sr', type=int)
parser.add_argument('--wav2_start_sr', type=int)
parser.add_argument('--wav1_end_sr', type=int)
parser.add_argument('--wav2_end_sr', type=int)
parser.add_argument('--wav2_cut_wav1', type=bool, default=False)
args = parser.parse_args()

tsv1 = open(args.tsv1)
tsv1_root = tsv1.readline().strip()
tsv1 = tsv1.readlines()

tsv2 = open(args.tsv2)
tsv2_root = tsv2.readline().strip()
tsv2 = tsv2.readlines()

scores_w = open(args.scores, 'w')

assert len(tsv1) == len(tsv2)

model = None
score_list = []
for t1, t2 in tqdm.tqdm(zip(tsv1, tsv2), total=len(tsv1)):
    t1_name = t1.split()[0]
    t2_name = t2.split()[0]
    try:
        print(f"processing {t1_name} {t2_name}")
        sim, model = verification(args.model_name,  tsv1_root+'/'+t1_name, tsv2_root+'/'+t2_name, use_gpu=True, checkpoint=args.checkpoint, wav1_start_sr=args.wav1_start_sr, wav2_start_sr=args.wav2_start_sr, wav1_end_sr=args.wav1_end_sr, wav2_end_sr=args.wav2_end_sr, model=model, wav2_cut_wav1=args.wav2_cut_wav1)
    Exception e:
        continue
    scores_w.write(f'{t1_name}_{args.wav1_start_sr}_{args.wav1_end_sr}|{t2_name}_{args.wav2_start_sr}_{args.wav2_end_sr}\t{sim.cpu().item()}\n')
    print(f'{t1_name}_{args.wav1_start_sr}_{args.wav1_end_sr}|{t2_name}_{args.wav2_start_sr}_{args.wav2_end_sr}\t{sim.cpu().item()}')
    score_list.append(sim.cpu().item())
    scores_w.flush()
print(f'avg score: {sum(score_list)/len(score_list)}')

