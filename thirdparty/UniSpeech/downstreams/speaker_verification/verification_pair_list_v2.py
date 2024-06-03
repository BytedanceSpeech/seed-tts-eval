'''
python verification_tsv.py $tsv1 $tsv2 --model_name wavlm_large --checkpoint wavlm_large_finetune.pth --scores $score_file --wav1_start_sr 0 --wav2_start_sr 0 --wav1_end_sr -1 --wav2_end_sr -1
'''
import tqdm
import argparse
from verification import verification
import os

parser = argparse.ArgumentParser()
parser.add_argument('pair')
parser.add_argument('--model_name')
parser.add_argument('--checkpoint')
parser.add_argument('--scores')
parser.add_argument('--wav1_start_sr', type=int)
parser.add_argument('--wav2_start_sr', type=int)
parser.add_argument('--wav1_end_sr', type=int)
parser.add_argument('--wav2_end_sr', type=int)
parser.add_argument('--wav2_cut_wav1', type=bool, default=False)
parser.add_argument('--device', default="cuda:0")
args = parser.parse_args()

f = open(args.pair)
lines = f.readlines()
f.close()

tsv1 = []
tsv2 = []
for line in lines:
    e = line.strip().split('|')
    if len(e) == 4:
        part1, _, _, part2 = line.strip().split('|')
    else:
        part1, part2 = line.strip().split('|')[:2]
    tsv1.append(part1)
    tsv2.append(part2)

scores_w = open(args.scores, 'w')
assert len(tsv1) == len(tsv2)

model = None
score_list = []
for t1, t2 in tqdm.tqdm(zip(tsv1, tsv2), total=len(tsv1)):
    t1_path = t1.strip()
    t2_path = t2.strip()
    if not os.path.exists(t1_path) or not os.path.exists(t2_path):
        continue
    try:
        sim, model = verification(args.model_name, t1_path, t2_path, use_gpu=True, checkpoint=args.checkpoint, wav1_start_sr=args.wav1_start_sr, wav2_start_sr=args.wav2_start_sr, wav1_end_sr=args.wav1_end_sr, wav2_end_sr=args.wav2_end_sr, model=model, wav2_cut_wav1=args.wav2_cut_wav1, device=args.device)
    except Exception as e:
        print(str(e))
        continue

    if sim is None:
        continue
    scores_w.write(f'{t1_path}_{args.wav1_start_sr}_{args.wav1_end_sr}|{t2_path}_{args.wav2_start_sr}_{args.wav2_end_sr}\t{sim.cpu().item()}\n')
    # print(f'{t1_path}_{args.wav1_start_sr}_{args.wav1_end_sr}|{t2_path}_{args.wav2_start_sr}_{args.wav2_end_sr}\t{sim.cpu().item()}')
    score_list.append(sim.cpu().item())
    scores_w.flush()
scores_w.write(f'avg score: {sum(score_list)/len(score_list)}')
scores_w.flush()
# print(f'avg score: {round(sum(score_list)/len(score_list), 3)}')
