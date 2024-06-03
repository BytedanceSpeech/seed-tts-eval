# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import librosa
import soundfile as sf

from pydub import AudioSegment


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wav-path', type=str
    )
    parser.add_argument(
        '--dest-path', type=str
    )
    parser.add_argument(
        '--input', type=str
    )
    parser.add_argument(
        '--output', type=str
    )

    return parser


def main(args):
    os.makedirs(args.dest_path, exist_ok=True)

    f = open(args.input)
    data = f.readlines()
    f.close()

    wf = open(args.output, 'w')
    wf.write(args.dest_path+"\n")
    count = len(data)
    for line in data:
        wav_name = line.strip()
        wav_file = os.path.join(args.wav_path, wav_name)
        base_wav_name = os.path.splitext(wav_name)[0]
        output_file = os.path.join(args.dest_path, base_wav_name+".wav")
        if os.path.exists(wav_file) and not os.path.exists(output_file):
            sound = AudioSegment.from_mp3(wav_file)
            sound.export(os.path.join(args.dest_path, 'tmp.wav'), format='wav')
            y, sr = librosa.load(os.path.join(args.dest_path, 'tmp.wav'), sr=16000)
            sf.write(output_file, y, sr)
            infos = sf.info(output_file)
            frames = infos.frames
            sr = infos.samplerate
            wf.write("{}\t{}\t{}\n".format(base_wav_name+".wav", frames, sr))
            count += 1
        if count % 100 == 0:
            print('process {} done'.format(count))

    wf.close()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
