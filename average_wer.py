import sys
import numpy as np

infile=sys.argv[1]
outfile=sys.argv[2]

fout = open(outfile, "w")
fout.write("utt" + '\t' + "wav_res" + '\t' + 'res_wer' + '\t' + 'text_ref' + '\t' + 'text_res' + '\t' + 'res_wer_ins' + '\t' + 'res_wer_del' + '\t' + 'res_wer_sub' + '\n')
wers = []
wers_below50 = []
inses = []
deles = []
subses = []
n_higher_than_50 = 0
for line in open(infile, "r").readlines():
    wav_path, wer, text_ref, text_res, inse, dele, subs = line.strip().split("\t")
    if float(wer) > 0.5:
        n_higher_than_50 += 1
    else:
        wers_below50.append(float(wer))
    wers.append(float(wer))
    inses.append(float(inse))
    deles.append(float(dele))
    subses.append(float(subs))
    fout.write(line)

wer = round(np.mean(wers)*100,3)
wer_below50 = round(np.mean(wers_below50)*100,3)
subs = round(np.mean(subses)*100,3)
dele = round(np.mean(deles)*100,3)
inse = round(np.mean(inses)*100,3)

subs_ratio = round(subs / wer, 3)
dele_ratio = round(dele / wer, 3)
inse_ratio = round(inse / wer, 3)

fout.write(f"WER: {wer}%\n")
fout.close()

print(f"WER: {wer}%\n")
