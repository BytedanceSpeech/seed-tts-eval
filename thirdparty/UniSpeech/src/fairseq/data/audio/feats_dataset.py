# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
import io

import pdb
import numpy as np
import torch
import torch.nn.functional as F
from fairseq.data import data_utils

from .. import FairseqDataset
from ..data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes
from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
    mulaw_encode,
    preemphasis,
)


logger = logging.getLogger(__name__)

class FeatsAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        input_feature="mfcc",
        output_feature="mfcc",
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=False,
            normalize=normalize
        )

        self.chunk_names = []
        self.chunk_indices = []
        self.fnames = []
        self.skipped = []
        self.speakers = []
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.speaker_dict = {}
        speaker_count = 0

        skipped = 0
        count = 0
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                #assert len(items) == 2, line
                sz = int(items[1])
                if self.input_feature != "wav":
                    sz = int(sz/self.sample_rate*100)
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped.append(i)
                    self.skipped_indices.add(i)
                    continue
                fname = items[0].split(":")
                if len(fname) > 1:
                    if len(self.chunk_names) == 0 or fname[0] != self.chunk_names[-1]:
                        self.chunk_names.append(fname[0])
                        self.chunk_indices.append(len(self.fnames))
                self.fnames.append(items[0])
                if len(items) > 2:
                    speaker = int(items[2])
                else:
                    speaker = int(items[0].split("/")[-1].split("-")[0])
                if speaker not in self.speaker_dict:
                    self.speaker_dict[speaker] = speaker_count
                    speaker_count += 1
                self.speakers.append(self.speaker_dict[speaker])
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass


    def get_mfcc(self, wav, sample_rate=16000, normalize=True):
        try:
            import torchaudio
            import torchaudio.compliance.kaldi as ta_kaldi
            with torch.no_grad():
                x = torch.from_numpy(wav).float()
                x = x.view(1, -1)

                mfccs = ta_kaldi.mfcc(
                    waveform=x,
                    sample_frequency=sample_rate,
                    use_energy=False,
                )  # (time, freq)
                mfccs = mfccs.transpose(0, 1)  # (freq, time)
                deltas = torchaudio.functional.compute_deltas(mfccs)
                ddeltas = torchaudio.functional.compute_deltas(deltas)
                concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
                concat = concat.transpose(0, 1).contiguous()  # (freq, time)
            if normalize:
                mean = concat.mean(dim=0)
                std = concat.std(dim=0)
                concat = (concat - mean) / std
            return concat
        except ImportError:
            return None

    def get_logmel(self, wav, sample_rate=16000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
            win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):
        wav = wav / np.abs(wav).max() * 0.999

        try:
            import librosa
            mel = librosa.feature.melspectrogram(preemphasis(wav, preemph),
                    sr=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length,
                    win_length=win_length, fmin=fmin, power=1)
            logmel = librosa.amplitude_to_db(mel, top_db=top_db)

            logmel = logmel / top_db + 1
            logmel = torch.from_numpy(logmel).transpose(0, 1)
            return logmel
        except ImportError:
            return None


    def get_fbank(self, wav, n_bins=80, sample_rate=16000, normalize=True):
        try:
            import torchaudio.compliance.kaldi as ta_kaldi
            x = torch.from_numpy(wav).float()
            x = x.view(1, -1)
            features = ta_kaldi.fbank(
                x, num_mel_bins=n_bins, sample_frequency=sample_rate
            )
            if normalize:
                mean = features.mean(dim=0)
                std = features.std(dim=0)
                features = (features - mean) / std
            return features
        except ImportError:
            return None

    def mulaw_encode(self, wav):
        wav = wav / np.abs(wav).max() * 0.999
        wav = mulaw_encode(wav, mu=2**8)
        return wav

    def __getitem__(self, index):
        import soundfile as sf

        path_or_fp = os.path.join(self.root_dir, str(self.fnames[index]))
        _path, slice_ptr = parse_path(path_or_fp)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)
        wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")

        if self.input_feature == "wav":
            feats = torch.from_numpy(wav).float()
            feats = self.postprocess(feats, curr_sample_rate)
        elif self.input_feature == "fbank":
            feats = self.get_fbank(wav, n_bins=80, sample_rate=curr_sample_rate)
        elif self.input_feature == "mfcc":
            feats = self.get_mfcc(wav, sample_rate=curr_sample_rate)
        elif self.input_feature == "logmel":
            feats = self.get_logmel(wav, sample_rate=curr_sample_rate)
        elif self.input_feature == "mulaw":
            feats = self.mulaw_encode(wav)
            feats = torch.from_numpy(feats).long()
        else:
            raise ValueError("Unknown extra features {}".format(self.input_feature))

        if self.output_feature == self.input_feature:
            target = feats
        elif self.output_feature == "wav":
            target = torch.from_numpy(wav).float()
            feats = self.postprocess(feats, curr_sample_rate)
        elif self.output_feature == "fbank":
            target = self.get_fbank(wav, n_bins=80, sample_rate=curr_sample_rate)
        elif self.output_feature == "mfcc":
            target = self.get_mfcc(wav, sample_rate=curr_sample_rate)
        elif self.output_feature == "logmel":
            target = self.get_logmel(wav, sample_rate=curr_sample_rate)
        elif self.output_feature == "mulaw":
            target = self.mulaw_encode(wav)
            target = torch.from_numpy(target).long()
        else:
            raise ValueError("Unknown extra features {}".format(self.output_feature))

        return {"id": index, "input": feats, "target": target, "speaker": self.speakers[index]}



    def collater(self, samples):
        samples = [s for s in samples if s["input"] is not None]
        if len(samples) == 0:
            return {}

        inputs = [s["input"] for s in samples]
        targets = [s["target"] for s in samples]
        sizes = [len(s) for s in inputs]
        speakers = [s['speaker'] for s in samples]

        input_size = min(min(sizes), self.max_sample_size)
        if input_size % 2 != 0:
            input_size = input_size - 1
        """
        if self.input_feature == "wav" or self.input_feature == "mulaw":
            if self.output_feature in ["mfcc", "fbank"]:
                target_rate = 1.0 / 160
            if self.output_feature == "logmel":
                target_rate = 1.0 / 160
                start_offset = -1
                end_offset = 1
        elif self.input_feature == "mfcc" or self.input_feature == "fbank":
            if self.output_feature not in ["mfcc", "fbank", "logmel"]:
                target_rate = 160
        elif self.input_feature == "logmel":
            if self.output_feature not in ["mfcc", "fbank", "logmel"]:
                target_rate = 160
        """
        if self.input_feature == self.output_feature:
            target_rate = 1
            offset = 0
        elif self.input_feature == "logmel" and self.output_feature =="mulaw":
            target_rate = 160
            offset = 1
        else:
            raise ValueError("Unsupport {} and {}".format(self.input_feature, self.output_feature))


        if inputs[0].dim() == 2:
            collated_inputs = inputs[0].new_zeros(len(inputs), input_size+offset*2, inputs[0].shape[-1])
        else:
            collated_inputs = inputs[0].new_zeros(len(inputs), input_size+offset*2)
        if targets[0].dim() == 2:
            collated_targets = targets[0].new_zeros(len(inputs), (input_size) * target_rate + offset, targets[0].shape[-1])
        else:
            collated_targets = targets[0].new_zeros(len(inputs), (input_size) * target_rate + offset)

        for i, (input, size) in enumerate(zip(inputs, sizes)):
            size = len(input)
            start = np.random.randint(offset, size - input_size + 1)
            collated_inputs[i] = input[start-offset: start + input_size + offset]
            collated_targets[i] = targets[i][start * target_rate: (start+input_size) * target_rate + offset]

         
        out = {"id": torch.LongTensor([s["id"] for s in samples]), "speakers":torch.LongTensor(speakers)}
        out["input"] = collated_inputs
        out["target"] = collated_targets

        return out
