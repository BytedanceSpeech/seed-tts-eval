# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ dataset.py ]
#   Synopsis     [ the speaker diarization dataset ]
#   Source       [ Refactored from https://github.com/hitachi-speech/EEND ]
#   Author       [ Jiatong Shi ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import io
import os
import subprocess
import sys

# -------------#
import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence

# -------------#
from torch.utils.data.dataset import Dataset
# -------------#


def _count_frames(data_len, size, step):
    # no padding at edges, last remaining samples are ignored
    return int((data_len - size + step) / step)


def _gen_frame_indices(data_length, size=2000, step=2000):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size

    if i * step + size < data_length:
        if data_length - (i + 1) * step > 0:
            if i == -1:
                yield (i + 1) * step, data_length
            else:
                yield data_length - size, data_length


def _gen_chunk_indices(data_len, chunk_size):
    step = chunk_size
    start = 0
    while start < data_len:
        end = min(data_len, start + chunk_size)
        yield start, end
        start += step


#######################
# Diarization Dataset #
#######################
class DiarizationDataset(Dataset):
    def __init__(
        self,
        mode,
        data_dir,
        chunk_size=2000,
        frame_shift=256,
        sampling_rate=16000,
        subsampling=1,
        use_last_samples=True,
        num_speakers=3,
        filter_spk=False
    ):
        super(DiarizationDataset, self).__init__()

        self.mode = mode
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.n_speakers = num_speakers
        self.chunk_indices = [] if mode != "test" else {}

        self.data = KaldiData(self.data_dir)
        self.all_speakers = sorted(self.data.spk2utt.keys())
        self.all_n_speakers = len(self.all_speakers)

        # make chunk indices: filepath, start_frame, end_frame
        for rec in self.data.wavs:
            data_len = int(self.data.reco2dur[rec] * sampling_rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            if mode == "test":
                self.chunk_indices[rec] = []
            if mode != "test":
                for st, ed in _gen_frame_indices(data_len, chunk_size, chunk_size):
                    self.chunk_indices.append(
                        (rec, st * self.subsampling, ed * self.subsampling)
                    )
            else:
                for st, ed in _gen_chunk_indices(data_len, chunk_size):
                    self.chunk_indices[rec].append(
                        (rec, st, ed)
                    )

        if mode != "test":
            if filter_spk:
                self.filter_spk()
            print(len(self.chunk_indices), " chunks")
        else:
            self.rec_list = list(self.chunk_indices.keys())
            print(len(self.rec_list), " recordings")

    def __len__(self):
        return (
            len(self.rec_list)
            if type(self.chunk_indices) == dict
            else len(self.chunk_indices)
        )

    def filter_spk(self):
        # filter the spk in spk2utt but will not be used in training
        # i.e. the chunks contains more spk than self.n_speakers
        occur_spk_set = set()

        new_chunk_indices = []  # filter the chunk that more than self.num_speakers
        for idx in range(self.__len__()):
            rec, st, ed = self.chunk_indices[idx]

            filtered_segments = self.data.segments[rec]
            # all the speakers in this recording not the chunk
            speakers = np.unique(
                [self.data.utt2spk[seg['utt']] for seg in filtered_segments]
                ).tolist()
            n_speakers = self.n_speakers
            # we assume that in each chunk the speaker number is less or equal than self.n_speakers
            # but the speaker number in the whole recording may exceed self.n_speakers
            if self.n_speakers < len(speakers):
                n_speakers = len(speakers)

            # Y: (length,), T: (frame_num, n_speakers)
            Y, T = self._get_labeled_speech(rec, st, ed, n_speakers)
            # the spk index exist in this chunk data
            exist_spk_idx = np.sum(T, axis=0) > 0.5  #  bool index
            chunk_spk_num = np.sum(exist_spk_idx)
            if chunk_spk_num <= self.n_speakers:
                spk_arr = np.array(speakers)
                valid_spk_arr = spk_arr[exist_spk_idx[:spk_arr.shape[0]]]
                for spk in valid_spk_arr:
                    occur_spk_set.add(spk)

                new_chunk_indices.append((rec, st, ed))
        self.chunk_indices = new_chunk_indices
        self.all_speakers = sorted(list(occur_spk_set))
        self.all_n_speakers = len(self.all_speakers)

    def __getitem__(self, i):
        if self.mode != "test":
            rec, st, ed = self.chunk_indices[i]

            filtered_segments = self.data.segments[rec]
            # all the speakers in this recording not the chunk
            speakers = np.unique(
                [self.data.utt2spk[seg['utt']] for seg in filtered_segments]
                ).tolist()
            n_speakers = self.n_speakers
            # we assume that in each chunk the speaker number is less or equal than self.n_speakers
            # but the speaker number in the whole recording may exceed self.n_speakers 
            if self.n_speakers < len(speakers):
                n_speakers = len(speakers)

            # Y: (length,), T: (frame_num, n_speakers)
            Y, T = self._get_labeled_speech(rec, st, ed, n_speakers)
            # the spk index exist in this chunk data
            exist_spk_idx = np.sum(T, axis=0) > 0.5  #  bool index
            chunk_spk_num = np.sum(exist_spk_idx)
            if chunk_spk_num > self.n_speakers:
                # the speaker number in a chunk exceed our pre-set value
                return None, None, None

            # the map from within recording speaker index to global speaker index
            S_arr = -1 * np.ones(n_speakers).astype(np.int64)
            for seg in filtered_segments:
                speaker_index = speakers.index(self.data.utt2spk[seg['utt']])
                try:
                    all_speaker_index = self.all_speakers.index(
                        self.data.utt2spk[seg['utt']])
                except:
                    #  we have pre-filter some spk in self.filter_spk
                    all_speaker_index = -1
                S_arr[speaker_index] = all_speaker_index
            # If T[:, n_speakers - 1] == 0.0, then S_arr[n_speakers - 1] == -1,
            # so S_arr[n_speakers - 1] is not used for training,
            # e.g., in the case of training 3-spk model with 2-spk data

            # filter the speaker not exist in this chunk and ensure there are self.num_speakers outputs
            T_exist = T[:,exist_spk_idx]
            T = np.zeros((T_exist.shape[0], self.n_speakers), dtype=np.int32)
            T[:,:T_exist.shape[1]] = T_exist
            #  subsampling for Y will be done in the model forward function
            T = T[::self.subsampling]

            S_arr_exist = S_arr[exist_spk_idx]
            S_arr = -1 * np.ones(self.n_speakers).astype(np.int64)
            S_arr[:S_arr_exist.shape[0]] = S_arr_exist

            n = np.arange(self.all_n_speakers, dtype=np.int64).reshape(self.all_n_speakers, 1)
            return Y, T, S_arr, n, T.shape[0]
        else:
            len_ratio = self.frame_shift * self.subsampling
            chunks = self.chunk_indices[self.rec_list[i]]
            Ys = []
            chunk_len_list = []
            for (rec, st, ed) in chunks:
                chunk_len = ed - st
                if chunk_len != self.chunk_size:
                    st = max(0, ed - self.chunk_size)
                Y, _ = self.data.load_wav(rec, st * len_ratio, ed * len_ratio)
                Ys.append(Y)
                chunk_len_list.append(chunk_len)
            return Ys, self.rec_list[i], chunk_len_list

    def get_allnspk(self):
        return self.all_n_speakers

    def _get_labeled_speech(
        self, rec, start, end, n_speakers=None, use_speaker_id=False
    ):
        """Extracts speech chunks and corresponding labels

        Extracts speech chunks and corresponding diarization labels for
        given recording id and start/end times

        Args:
            rec (str): recording id
            start (int): start frame index
            end (int): end frame index
            n_speakers (int): number of speakers
                if None, the value is given from data
        Returns:
            data: speech chunk
                (n_samples)
            T: label
                (n_frmaes, n_speakers)-shaped np.int32 array.
        """
        data, rate = self.data.load_wav(
            rec, start * self.frame_shift, end * self.frame_shift
        )
        frame_num = end - start
        filtered_segments = self.data.segments[rec]
        # filtered_segments = self.data.segments[self.data.segments['rec'] == rec]
        speakers = np.unique(
            [self.data.utt2spk[seg["utt"]] for seg in filtered_segments]
        ).tolist()
        if n_speakers is None:
            n_speakers = len(speakers)
        T = np.zeros((frame_num, n_speakers), dtype=np.int32)

        if use_speaker_id:
            all_speakers = sorted(self.data.spk2utt.keys())
            S = np.zeros((frame_num, len(all_speakers)), dtype=np.int32)

        for seg in filtered_segments:
            speaker_index = speakers.index(self.data.utt2spk[seg["utt"]])
            if use_speaker_id:
                all_speaker_index = all_speakers.index(self.data.utt2spk[seg["utt"]])
            start_frame = np.rint(seg["st"] * rate / self.frame_shift).astype(int)
            end_frame = np.rint(seg["et"] * rate / self.frame_shift).astype(int)
            rel_start = rel_end = None
            if start <= start_frame and start_frame < end:
                rel_start = start_frame - start
            if start < end_frame and end_frame <= end:
                rel_end = end_frame - start
            if rel_start is not None or rel_end is not None:
                T[rel_start:rel_end, speaker_index] = 1
                if use_speaker_id:
                    S[rel_start:rel_end, all_speaker_index] = 1

        if use_speaker_id:
            return data, T, S
        else:
            return data, T

    def collate_fn(self, batch):
        valid_samples = [sample for sample in batch if sample[0] is not None]

        wav_list, binary_label_list, spk_label_list= [], [], []
        all_spk_idx_list, len_list = [], []
        for sample in valid_samples:
            wav_list.append(torch.from_numpy(sample[0]).float())
            binary_label_list.append(torch.from_numpy(sample[1]).long())
            spk_label_list.append(torch.from_numpy(sample[2]).long())
            all_spk_idx_list.append(torch.from_numpy(sample[3]).long())
            len_list.append(sample[4])
        wav_batch = pad_sequence(wav_list, batch_first=True, padding_value=0.0)
        binary_label_batch = pad_sequence(binary_label_list, batch_first=True, padding_value=1).long()
        spk_label_batch = torch.stack(spk_label_list)
        all_spk_idx_batch = torch.stack(all_spk_idx_list)
        len_batch = torch.LongTensor(len_list)

        return wav_batch, binary_label_batch.float(), spk_label_batch, all_spk_idx_batch, len_batch

    def collate_fn_infer(self, batch):
        assert len(batch) == 1  # each batch should contain one recording
        Ys, rec, chunk_len_list = batch[0]
        wav_list = [torch.from_numpy(Y).float() for Y in Ys]

        return wav_list, rec, chunk_len_list


#######################
# Kaldi-style Dataset #
#######################
class KaldiData:
    """This class holds data in kaldi-style directory."""

    def __init__(self, data_dir):
        """Load kaldi data directory."""
        self.data_dir = data_dir
        self.segments = self._load_segments_rechash(
            os.path.join(self.data_dir, "segments")
        )
        self.utt2spk = self._load_utt2spk(os.path.join(self.data_dir, "utt2spk"))
        self.wavs = self._load_wav_scp(os.path.join(self.data_dir, "wav.scp"))
        self.reco2dur = self._load_reco2dur(os.path.join(self.data_dir, "reco2dur"))
        self.spk2utt = self._load_spk2utt(os.path.join(self.data_dir, "spk2utt"))

    def load_wav(self, recid, start=0, end=None):
        """Load wavfile given recid, start time and end time."""
        data, rate = self._load_wav(self.wavs[recid], start, end)
        return data, rate

    def _load_segments(self, segments_file):
        """Load segments file as array."""
        if not os.path.exists(segments_file):
            return None
        return np.loadtxt(
            segments_file,
            dtype=[("utt", "object"), ("rec", "object"), ("st", "f"), ("et", "f")],
            ndmin=1,
        )

    def _load_segments_hash(self, segments_file):
        """Load segments file as dict with uttid index."""
        ret = {}
        if not os.path.exists(segments_file):
            return None
        for line in open(segments_file):
            utt, rec, st, et = line.strip().split()
            ret[utt] = (rec, float(st), float(et))
        return ret

    def _load_segments_rechash(self, segments_file):
        """Load segments file as dict with recid index."""
        ret = {}
        if not os.path.exists(segments_file):
            return None
        for line in open(segments_file):
            utt, rec, st, et = line.strip().split()
            if rec not in ret:
                ret[rec] = []
            ret[rec].append({"utt": utt, "st": float(st), "et": float(et)})
        return ret

    def _load_wav_scp(self, wav_scp_file):
        """Return dictionary { rec: wav_rxfilename }."""
        if os.path.exists(wav_scp_file):
            lines = [line.strip().split(None, 1) for line in open(wav_scp_file)]
            return {x[0]: x[1] for x in lines}
        else:
            wav_dir = os.path.join(self.data_dir, "wav")
            return {
                os.path.splitext(filename)[0]: os.path.join(wav_dir, filename)
                for filename in sorted(os.listdir(wav_dir))
            }

    def _load_wav(self, wav_rxfilename, start=0, end=None):
        """This function reads audio file and return data in numpy.float32 array.
        "lru_cache" holds recently loaded audio so that can be called
        many times on the same audio file.
        OPTIMIZE: controls lru_cache size for random access,
        considering memory size
        """
        if wav_rxfilename.endswith("|"):
            # input piped command
            p = subprocess.Popen(
                wav_rxfilename[:-1],
                shell=True,
                stdout=subprocess.PIPE,
            )
            data, samplerate = sf.read(
                io.BytesIO(p.stdout.read()),
                dtype="float32",
            )
            # cannot seek
            data = data[start:end]
        elif wav_rxfilename == "-":
            # stdin
            data, samplerate = sf.read(sys.stdin, dtype="float32")
            # cannot seek
            data = data[start:end]
        else:
            # normal wav file
            data, samplerate = sf.read(wav_rxfilename, start=start, stop=end)
        return data, samplerate

    def _load_utt2spk(self, utt2spk_file):
        """Returns dictionary { uttid: spkid }."""
        lines = [line.strip().split(None, 1) for line in open(utt2spk_file)]
        return {x[0]: x[1] for x in lines}

    def _load_spk2utt(self, spk2utt_file):
        """Returns dictionary { spkid: list of uttids }."""
        if not os.path.exists(spk2utt_file):
            return None
        lines = [line.strip().split() for line in open(spk2utt_file)]
        return {x[0]: x[1:] for x in lines}

    def _load_reco2dur(self, reco2dur_file):
        """Returns dictionary { recid: duration }."""
        if not os.path.exists(reco2dur_file):
            return None
        lines = [line.strip().split(None, 1) for line in open(reco2dur_file)]
        return {x[0]: float(x[1]) for x in lines}

    def _process_wav(self, wav_rxfilename, process):
        """This function returns preprocessed wav_rxfilename.
        Args:
            wav_rxfilename:
                input
            process:
                command which can be connected via pipe, use stdin and stdout
        Returns:
            wav_rxfilename: output piped command
        """
        if wav_rxfilename.endswith("|"):
            # input piped command
            return wav_rxfilename + process + "|"
        # stdin "-" or normal file
        return "cat {0} | {1} |".format(wav_rxfilename, process)

    def _extract_segments(self, wavs, segments=None):
        """This function returns generator of segmented audio.
        Yields (utterance id, numpy.float32 array).
        TODO?: sampling rate is not converted.
        """
        if segments is not None:
            # segments should be sorted by rec-id
            for seg in segments:
                wav = wavs[seg["rec"]]
                data, samplerate = self.load_wav(wav)
                st_sample = np.rint(seg["st"] * samplerate).astype(int)
                et_sample = np.rint(seg["et"] * samplerate).astype(int)
                yield seg["utt"], data[st_sample:et_sample]
        else:
            # segments file not found,
            # wav.scp is used as segmented audio list
            for rec in wavs:
                data, samplerate = self.load_wav(wavs[rec])
                yield rec, data

if __name__ == "__main__":
    args = {
        'mode': 'train',
        'data_dir': "/mnt/lustre/sjtu/home/czy97/workspace/sd/EEND-vec-clustering/EEND-vector-clustering/egs/mini_librispeech/v1/data/simu/data/train_clean_5_ns3_beta2_500",
        'chunk_size': 2001,
        'frame_shift': 256,
        'sampling_rate': 8000,
        'num_speakers':3
    }

    torch.manual_seed(6)
    dataset = DiarizationDataset(**args)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
    data_iter = iter(dataloader)
    # wav_batch, binary_label_batch, spk_label_batch, all_spk_idx_batch, len_batch = next(data_iter)
    data = next(data_iter)
    for val in data:
        print(val.shape)

    # from torch.utils.data import DataLoader
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn_infer)
    # data_iter = iter(dataloader)
    # wav_list, binary_label_list, rec = next(data_iter)
