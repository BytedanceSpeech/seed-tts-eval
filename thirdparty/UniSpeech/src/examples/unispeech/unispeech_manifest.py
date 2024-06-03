# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        'input', 
        type=str,
        help="input .tsv file"
    )
    parser.add_argument(
        '--dest',
        type=str,
        help="output directory"
    )

    return parser


def main(args):
    wav_names = []
    text = []
    with open(args.input) as f:
        f.readline()
        for line in f:
            items = line.strip().split("\t")
            wav_names.append(items[1])
            text.append(items[2])
    base_name = os.path.basename(args.input)
    file_name = os.path.splitext(base_name)[0]

    with open(os.path.join(args.dest, file_name+'.list'), 'w') as f:
        for name in wav_names:
            f.write(name+"\n")

    with open(os.path.join(args.dest, file_name+'.text'), 'w') as f:
        for i in range(len(wav_names)):
            f.write("{}\t{}\n".format(wav_names[i], text[i]))

	
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
