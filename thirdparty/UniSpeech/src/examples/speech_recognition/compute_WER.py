# encoding: utf-8

import editdistance
import argparse
import re


def compute_wer_txt(hypo_f, ref_f):
    f = open(hypo_f)
    hypos = f.readlines()
    f.close()
    f = open(ref_f)
    refs = f.readlines()
    f.close()
    err,cnt = 0,0
    print("Start Computing WER")
    for i in range(len(hypos)):
        hypo = hypos[i].strip().split()[:-1]
        ref = refs[i].strip().split()[:-1]

        e = editdistance.eval(ref, hypo)
        #print(ref_token,predict_token,e/float(len(ref_token)+0.001))

        err += e
        cnt += len(ref)
    print(err,cnt,float(float(err+0.0) / float(cnt+0.01)*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypo', '--h', type=str, required=True,
                help='Input file path')
    parser.add_argument('--ref', '--r', type=str, required=True)

    args = parser.parse_args()
    compute_wer_txt(args.hypo, args.ref)
