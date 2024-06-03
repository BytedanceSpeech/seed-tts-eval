import sys, os
from tqdm import tqdm

metalst = sys.argv[1]
wav_dir = sys.argv[2]
wav_res_ref_text = sys.argv[3]

f = open(metalst)
lines = f.readlines()
f.close()

f_w = open(wav_res_ref_text, 'w')
for line in tqdm(lines):
    if len(line.strip().split('|')) == 5:
        utt, prompt_text, prompt_wav, infer_text, infer_wav = line.strip().split('|')
    elif len(line.strip().split('|')) == 4:
        utt, prompt_text, prompt_wav, infer_text = line.strip().split('|')
    elif len(line.strip().split('|')) == 2:
        utt, infer_text = line.strip().split('|')
    elif len(line.strip().split('|')) == 3:
        utt, infer_text, prompt_wav = line.strip().split('|')
        if utt.endswith(".wav"):
            utt = utt[:-4]
    if not os.path.exists(os.path.join(wav_dir, utt + '.wav')):
        continue

    # tmp
    #prompt_wav = infer_wav

    if not os.path.isabs(prompt_wav):
        prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

    # if not os.path.isabs(infer_wav):
    #     infer_wav = os.path.join(os.path.dirname(metalst), infer_wav)

    if len(line.strip().split('|')) == 2:
        out_line = '|'.join([os.path.join(wav_dir, utt + '.wav'), infer_text])
    else:
        out_line = '|'.join([os.path.join(wav_dir, utt + '.wav'), prompt_wav, infer_text])
    f_w.write(out_line + '\n')
f_w.close()
