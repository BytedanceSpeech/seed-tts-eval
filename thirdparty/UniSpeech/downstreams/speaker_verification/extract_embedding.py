import argparse
from verification import extract_embedding
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict



parser = argparse.ArgumentParser()
parser.add_argument("--infile", default="/mnt/bn/jcong5/pretrain_data/process_rp_prompt/thread-00.lst")
parser.add_argument("--outdir", default="/mnt/bn/jcong5/pretrain_data/rp_prompt_embedding/")
parser.add_argument('--checkpoint', default="/mnt/bn/jcong5/workspace/bigtts-eval/.cache_dir/wavlm_large_finetune.pth")
parser.add_argument('--device', default="cuda:0")
args = parser.parse_args()

model = None

lines = open(args.infile, "r").readlines()
os.makedirs(args.outdir, exist_ok=True)

for i, item in enumerate(tqdm(lines)):
    wavpath = item.strip()
    utt = os.path.splitext(os.path.basename(wavpath))[0]
    output_path = os.path.join(args.outdir, utt+".npy")
    print(output_path)
    if os.path.exists(output_path):
        print("skip")
        continue
    sim, model = extract_embedding("wavlm_large", 
                                   wavpath, 
                                   use_gpu=True, 
                                   checkpoint=args.checkpoint, 
                                   model=model,
                                   device=args.device)
    np.save(os.path.join(args.outdir, utt), sim[0].cpu())