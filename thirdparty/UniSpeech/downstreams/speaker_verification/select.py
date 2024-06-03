import argparse
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# 生成随机样本数据

# 对样本数据进行 T-SNE 降维


# 根据 T-SNE 降维结果画图



parser = argparse.ArgumentParser()
parser.add_argument("--indir", default="/mnt/bn/jcong5/pretrain_data/rp_prompt_embedding/")
# parser.add_argument("--indir", default="/mnt/bn/jcong5/pretrain_data/process_rp_prompt/threshold-0.5/uniq_embeddings")
parser.add_argument('--checkpoint', default="/mnt/bn/jcong5/workspace/bigtts-eval/.cache_dir/wavlm_large_finetune.pth")
parser.add_argument('--output_dir', default="/mnt/bn/jcong5/pretrain_data/process_rp_prompt/")
parser.add_argument('--threshold', type=float, default=0.5)
args = parser.parse_args()

threshold=args.threshold
output_dir = os.path.join(args.output_dir, f"threshold-{threshold}")
os.makedirs(output_dir, exist_ok=True)
outfile=os.path.join(output_dir, "uniq.lst")
fout=open(outfile, "w")
repeat_out=open(os.path.join(output_dir, "repeat.lst"), "w")

model = None
sims = []
id2utt = {}
i=0
for item in tqdm(os.listdir(args.indir)):
    wavpath = os.path.join(args.indir, item)
    try:
        sim = np.load(wavpath, allow_pickle=True)
    except:
        print(f"skip-{wavpath}")
        continue
    sims.append(sim)
    id2utt[i] = wavpath
    i=i+1

sims = np.array(sims)
sims = cosine_similarity(sims)

indices = np.arange(sim.shape[0])
delete = []
delete_pair = defaultdict(list)
for i in range(sims.shape[0]):
    if i in delete:
        continue
    for j in range(i+1, sims.shape[0]):
        if sims[i, j] >= threshold:
            delete.append(j)
            delete_pair[i].append(j)

for i in range(sims.shape[0]):
    if i in delete:
        continue
    fout.write(id2utt[i]+"\n")

for k, v in delete_pair.items():
    repeat_out.write(id2utt[k]+"\n")
    for i in v:
        repeat_out.write(id2utt[i]+"\n")
    repeat_out.write("\n")




smis = np.array(sims)
# 对样本数据进行 T-SNE 降维
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(smis)
# 根据 T-SNE 降维结果画图
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.savefig("test2.png")