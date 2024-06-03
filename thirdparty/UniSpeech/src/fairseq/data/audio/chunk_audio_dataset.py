# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb
import logging
import os
import sys
import json

import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F

from .. import FairseqDataset, data_utils
from fairseq.tokenizer import char_tokenizer
from fairseq.data.audio.audio_utils import _group_to_batches_by_frames, _group_to_batches_by_utters, _group_to_batches_by_frame_x_label, DataParser

ENDIAN = 'little'

logger = logging.getLogger(__name__)


class ChunkAudioDataset(torch.utils.data.IterableDataset, FairseqDataset):
    def __init__(
        self,
        chunk_data_file,
        chunk_data_path=None,
        chunk_trans_path=None,
        max_sample_size=None,
        min_sample_size=None,
        max_tokens=None,
        pad=False,
        normalize=False,
        subset=None,
        shuffle=True,
        shard=True,
        label=False,
        dictionary=None,
        feature="audio",
        mean_file=None,
        invstd_file=None,
        batch_criterion="frame"
    ):
        self._data_path = chunk_data_path
        self._data_file = chunk_data_file
        self._trans_path = chunk_trans_path
        self.max_sample_size = (
                max_sample_size if max_sample_size is not None else sys.maxsize
                )
        self.min_sample_size = min_sample_size
        self.max_tokens = max_tokens
        self.pad = pad
        self.shuffle = shuffle
        self.shard = shard
        self.normalize = normalize
        self.label = label
        self.dictionary=dictionary
        self.feature = feature

        if mean_file is not None:
            self.mean = np.fromfile(mean_file, sep='\n')
        else:
            self.mean = None
        
        if invstd_file is not None:
            self.invstd = np.fromfile(invstd_file, sep='\n')
        else:
            self.invstd = None
        

        with open(self._data_file) as f:
            self._chunk_list = json.load(f)['fileInfo']

        if self._data_path is None:
            self._data_path = os.path.dirname(self._data_file)
        if self._trans_path is None:
            self._trans_path = os.path.dirname(self._data_file)

        self._chunk_num = len(self._chunk_list)
        self._example_num = 0
        self._dist_size = 1
        self._dist_rank = 0
        self.end_of_epoch = False
        for chunk in self._chunk_list:
            self._example_num += int(chunk['count'])

        logger.info(f"Open dataset {self._data_file}, total example count {self._example_num}")

        self.subset = subset
        self.parser = DataParser()
        self._buffer_size = 3000
        self._batch_criterion = batch_criterion
        self._example_buffer = []
        self._batch_buffer = []
        self._first_iteration = True
        self.iterable = None

    def __len__(self):
        return self._example_num

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            offset = self._dist_rank
            skip = self._dist_size
        else:
            offset = self._dist_size * worker_info.id + self._dist_rank
            skip = self._dist_size * worker_info.num_workers
        #print(self._chunk_list[13])
        if self.shard:
            self._sharded_list = list(self._chunk_list[offset::skip])
            value = len(self._chunk_list) % self._dist_size
            if value !=0 and self._dist_rank >= value:
                if worker_info is None or worker_info.id == worker_info.num_workers - 1:
                    np.random.seed(self._dist_rank)
                    pad_chunk = np.random.choice(self._chunk_list)
                    self._sharded_list.append(pad_chunk)
        else:
            self._sharded_list = self._chunk_list
        self.iterable = iter(self._chunk_deserializer())
        #print("{}/{} worker init in gpu {}, sharded data {}/{}".format(worker_info.id, worker_info.num_workers, self._dist_rank, len(self._sharded_list), len(self._chunk_list)))
        return self

    def reset(self, world_size=1, world_rank=0):
        #print("Reset Dataset")
        self._example_buffer = []
        self._batch_buffer = []
        self._first_iteration = True
        self._dist_size = world_size
        self._dist_rank = world_rank
        np.random.seed(self.epoch)
        if self.shuffle:
            np.random.shuffle(self._chunk_list)

    def set_epoch(self, epoch):
        self.epoch = epoch

    
    def __next__(self):
        return self._dynamicbatcher()

    def _read_chunk(self, file_path, chunk_type, chunk_size):
        example_list = []
        with open(file_path, 'rb') as f:
            target_type = f.read(len(chunk_type.encode())).decode()
            if chunk_type.lower() != target_type.lower():
                raise ValueError(
                        'Taget type is not expected in {}, expected {}, but got {}'
                        .format(file_path, chunk_type, target_type))
            version_number = int.from_bytes(f.read(4), byteorder=ENDIAN)

            for i in range(chunk_size):
                example_index = int.from_bytes(f.read(4), byteorder=ENDIAN)
                if example_index != i:
                    raise ValueError(
                            'The example index is corrupted in {}, \
                                    expected {}, but got {}'.format(
                                    file_path, i, example_index))
                data_size = int.from_bytes(f.read(4), byteorder=ENDIAN)
                data = f.read(data_size)
                example_list.append(data)
        return example_list

    def _chunk_deserializer(self):
        try:
            iterator = iter(self._sharded_list)
            chunk = next(iterator)
            while True:
                chunk_type = ['info', self.feature]
                if self.label:
                    chunk_type.append('transcription')
                chunk_name = chunk['name']
                chunk_size = int(chunk['count'])

                example_dict = {}
                for extension in chunk_type:
                    if extension == 'transcription':
                        file_path = os.path.join(self._trans_path, chunk_name+'.transcription')
                    else:
                        file_path = os.path.join(self._data_path, chunk_name+'.'+extension)

                    example_dict[extension] = self._read_chunk(file_path, extension, chunk_size)

                example_lens = [len(example_dict[x]) for x in chunk_type]
                if not all(x == chunk_size for x in example_lens):
                    error_msg = 'Chunk size is not consistent in chunk {}'.format(chunk_name)
                    raise ValueError(error_msg)

                for i in range(chunk_size):
                    one_example = {}
                    for extension in chunk_type:
                        one_example[extension] = self.parser._parse_data(example_dict[extension][i], extension)
                        if self.subset is not None and self.subset not in one_example['info']['corpusname']:
                            break
                    if 'transcription' in one_example:
                        one_example['y'] = self.dictionary.encode_line(
                            one_example['transcription'].upper(), line_tokenizer=char_tokenizer,
                            add_if_not_exist=False,
                            append_eos=False
                        )
                    if self.feature not in one_example:
                        continue

                    yield one_example
                chunk = next(iterator)
        except StopIteration:
            return

    def _fill_buffer_by_length(self, buffer, length):
        try:
            i = 0
            while i < length:
                example = next(self.iterable)
                x_len = example[self.feature].shape[0]
                if self.pad and self.max_sample_size is not None and x_len > self.max_sample_size:
                    continue
                if self.min_sample_size is not None and x_len < self.min_sample_size:
                    continue
                buffer.append(example)
                i += 1
        except StopIteration:
            pass

    def _create_batch_list(self, example_list):
        idx_len_pair = []
        for idx in range(len(example_list)):
            uttlen = len(example_list[idx][self.feature])
            if 'y' in example_list[idx]:
                target_len = len(example_list[idx]['y'])
            else:
                target_len = 1
            idx_len_pair.append((idx, uttlen, target_len))

        sorted_idx_len_pair = sorted(idx_len_pair, key=lambda var: var[1], reverse=self.pad)
        if self._batch_criterion == "frame":
            group_batches_fn = _group_to_batches_by_frames
        elif self._batch_criterion == "utterance":
            group_batches_fn = _group_to_batches_by_utters
        elif self._batch_criterion == "frame_x_label":
            group_batches_fn = _group_to_batches_by_frame_x_label
        else:
            raise ValueError("Only support for grouping batches by 'frame', 'utterance', 'frame_x_label'")

        batch_list = group_batches_fn(
                self._example_buffer, sorted_idx_len_pair, self.max_tokens) 
        if self.shuffle:
            np.random.shuffle(batch_list)
        return batch_list

    def _dynamicbatcher(self):
        if self._first_iteration:
            self._first_iteration = False
            self._fill_buffer_by_length(self._example_buffer, self._buffer_size)
            if self.shuffle:
                np.random.shuffle(self._example_buffer)

        if not self._batch_buffer and not self._example_buffer:
            raise StopIteration

        if not self._batch_buffer:
            self._batch_buffer = self._create_batch_list(self._example_buffer)
            self._example_buffer = []

        single_batch = self._batch_buffer.pop()
        self._fill_buffer_by_length(self._example_buffer, len(single_batch))
        if self.feature == "audio":
            sources = [self.postprocess(torch.from_numpy(s[self.feature])).float() for s in single_batch]
        else:
            sources = [torch.from_numpy(self.mvn(s[self.feature])).float() for s in single_batch]
        infos = [s['info'] for s in single_batch]
        ids = torch.LongTensor(list(range(len(single_batch))))
        if self.label:
            target = [s['y'] for s in single_batch]
            return {'info': infos, 'id': ids,'source': sources, "target": target}
        return {'info': infos, 'id': ids,'source': sources}

    def collater(self, samples):
        samples = samples[0]
        if len(samples["source"]) == 0:
            return {}

        sources = samples['source']
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        if self.feature == "audio":
            collated_sources = sources[0].new_zeros(len(sources), target_size)
        else:
            collated_sources = sources[0].new_zeros(len(sources), target_size, 80)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                if self.feature == "audio":
                    collated_sources[i] = torch.cat(
                        [source, source.new_full((-diff,), 0.0)]
                    )
                else:
                    collated_sources[i] = torch.cat(
                        [source, source.new_full((-diff, 80), 0.0)]
                    )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        collated = {"info": samples["info"], "id": samples["id"], "net_input": input}
        if not self.label:
            return collated
        target = samples['target']
        collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
        target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
        collated["ntokens"] = collated["target_lengths"].sum().item()
        collated["target"] = target
        return collated
           
    def postprocess(self, feats):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def mvn(self, feats):
        feats = (feats - self.mean) * self.invstd
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff+1)
        end = size - diff + start
        return wav[start:end]
