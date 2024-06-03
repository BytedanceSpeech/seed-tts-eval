'''
python extract_spks_from_score_file.py --scores $score_file --spks +0.9.12.18.21.34.64.71.87.92.99
python extract_spks_from_score_file.py --scores $score_file --spks +p225.p234.p238.p245.p248.p261.p294.p302.p326.p335.p347
'''
import tqdm
import argparse
from verification import verification

parser = argparse.ArgumentParser()
parser.add_argument('--scores')
parser.add_argument('--spks')
args = parser.parse_args()

scores = open(args.scores)
scores_w = open(args.scores+'.'+args.spks, 'w')

is_in = args.spks[0] == '+'
spks = args.spks[1:].split('.')
#spks = [int(i) for i in args.spks.split('.')]

score_list = []
for line in scores:
    if (is_in and line.split('_')[0].split('/')[0] in spks) or (not is_in and line.split('_')[0].split('/')[0] not in spks):
        scores_w.write(line)
        score_list.append(float(line.split()[-1]))
print(f'avg score: {sum(score_list)}/{len(score_list)}={sum(score_list)/len(score_list)}')
