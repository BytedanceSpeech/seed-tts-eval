# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys
import io

import numpy as np
import torch
import torch.nn.functional as F
from fairseq.data import data_utils

from .. import FairseqDataset
from ..data_utils import compute_mask_indices, get_buckets, get_bucketed_sizes
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)


logger = logging.getLogger(__name__)


class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        compute_mask_indices=False,
        **mask_compute_kwargs,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize
        self.compute_mask_indices = compute_mask_indices
        self.epoch = 0
        if self.compute_mask_indices:
            self.mask_compute_kwargs = mask_compute_kwargs
            self._features_size_map = {}
            self._C = mask_compute_kwargs["encoder_embed_dim"]
            self._conv_feature_layers = eval(mask_compute_kwargs["conv_feature_layers"])

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0
        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end], start

    def _compute_mask_indices(self, dims, padding_mask):
        B, T, C = dims
        mask_indices, mask_channel_indices = None, None
        if self.mask_compute_kwargs["mask_prob"] > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_compute_kwargs["mask_prob"],
                self.mask_compute_kwargs["mask_length"],
                self.mask_compute_kwargs["mask_selection"],
                self.mask_compute_kwargs["mask_other"],
                min_masks=2,
                no_overlap=self.mask_compute_kwargs["no_mask_overlap"],
                min_space=self.mask_compute_kwargs["mask_min_space"],
            )
            mask_indices = torch.from_numpy(mask_indices)
        if self.mask_compute_kwargs["mask_channel_prob"] > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_compute_kwargs["mask_channel_prob"],
                self.mask_compute_kwargs["mask_channel_length"],
                self.mask_compute_kwargs["mask_channel_selection"],
                self.mask_compute_kwargs["mask_channel_other"],
                no_overlap=self.mask_compute_kwargs["no_mask_channel_overlap"],
                min_space=self.mask_compute_kwargs["mask_channel_min_space"],
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).unsqueeze(1).expand(-1, T, -1)
            )

        return mask_indices, mask_channel_indices

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value=value)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i], start = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        
        out = {"id": torch.LongTensor([s["id"] for s in samples])}
        if self.pad:
            input["padding_mask"] = padding_mask

        if hasattr(self, "num_buckets") and self.num_buckets > 0:
            assert self.pad, "Cannot bucket without padding first."
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)
            num_pad = bucket - collated_sources.size(-1)
            if num_pad:
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)

        if self.compute_mask_indices:
            B = input["source"].size(0)
            T = self._get_mask_indices_dims(input["source"].size(-1))
            padding_mask_reshaped = input["padding_mask"].clone()
            extra = padding_mask_reshaped.size(1) % T
            if extra > 0:
                padding_mask_reshaped = padding_mask_reshaped[:, :-extra]
            padding_mask_reshaped = padding_mask_reshaped.view(
                padding_mask_reshaped.size(0), T, -1
            )
            padding_mask_reshaped = padding_mask_reshaped.all(-1)
            input["padding_count"] = padding_mask_reshaped.sum(-1).max().item()
            mask_indices, mask_channel_indices = self._compute_mask_indices(
                (B, T, self._C),
                padding_mask_reshaped,
            )
            input["mask_indices"] = mask_indices
            input["mask_channel_indices"] = mask_channel_indices
            out["sample_size"] = mask_indices.sum().item()

        out["net_input"] = input
        return out

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        if size not in self._features_size_map:
            L_in = size
            for (_, kernel_size, stride) in self._conv_feature_layers:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            if len(self.chunk_names) > 0:
                with data_utils.numpy_seed(self.epoch):
                    self.chunk_order = np.random.permutation(len(self.chunk_names))
                chunk_count = 0
                tmp_sizes = []
                tmp_indices = []
                indice = []
                for i in self.chunk_order:
                    chunk_count += 1
                    start = self.chunk_indices[i]
                    end = self.chunk_indices[i+1] if i < len(self.chunk_names) - 1 else len(self)
                    size = list(self.sizes[start:end])
                    tmp_indices.extend(list(np.arange(start, end)))
                    tmp_sizes.extend(size)
                    if chunk_count % 10 == 0 or i == self.chunk_order[0]:
                        order = [np.random.permutation(len(tmp_indices))]
                        order.append(
                            np.minimum(
                                np.array(tmp_sizes),
                                self.max_sample_size,
                            )
                        )
                        sort_idx = np.lexsort(order)[::-1]
                        indice.append([tmp_indices[k] for k in sort_idx])
                        tmp_indices = []
                        tmp_sizes =[]
                return indice
            else:
                order = [np.random.permutation(len(self))]
                order.append(
                    np.minimum(
                        np.array(self.sizes),
                        self.max_sample_size,
                    )
                )
                return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.required_batch_size_multiple = required_batch_size_multiple
        if isinstance(indices[0], list):
            batch_list = []
            for indice in indices:
                batch = super(RawAudioDataset, self).batch_by_size(indice, max_tokens, max_sentences, required_batch_size_multiple)
                batch_list.append(batch)
            return batch_list
        else:
            return super(RawAudioDataset, self).batch_by_size(indices, max_tokens, max_sentences, required_batch_size_multiple)

    def shuffle_batches(self, batches, seed):
        if isinstance(batches[0], list):
            new_batches = []
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
                for batch in batches:
                    np.random.shuffle(batch)
                    new_batches.extend(batch)
            return new_batches
        else:
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
        return batches

    def reset_batch_sampler(self):
        indices = self.ordered_indices()
        batch_sampler = self.batch_by_size(
                indices,
                self.max_tokens,
                self.max_sentences,
                self.required_batch_size_multiple
        )
        return batch_sampler

    def set_bucket_info(self, num_buckets):
        self.num_buckets = num_buckets
        if self.num_buckets > 0:
            self._collated_sizes = np.minimum(
                np.array(self.sizes),
                self.max_sample_size,
            )
            self.buckets = get_buckets(
                self._collated_sizes,
                self.num_buckets,
            )
            self._bucketed_sizes = get_bucketed_sizes(
                self._collated_sizes, self.buckets
            )
            logger.info(
                f"{len(self.buckets)} bucket(s) for the audio dataset: "
                f"{self.buckets}"
            )


class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        num_buckets=0,
        compute_mask_indices=False,
        **mask_compute_kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            compute_mask_indices=compute_mask_indices,
            **mask_compute_kwargs,
        )

        self.chunk_names = []
        self.chunk_indices = []
        self.fnames = []
        self.skipped = []

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
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped.append(i)
                    self.skipped_indices.add(i)
                    continue
                if pad and max_sample_size is not None and sz > max_sample_size:
                    skipped += 1
                    self.skipped.append(i)
                    continue
                fname = items[0].split(":")
                if len(fname) > 1:
                    if len(self.chunk_names) == 0 or fname[0] != self.chunk_names[-1]:
                        self.chunk_names.append(fname[0])
                        self.chunk_indices.append(len(self.fnames))
                self.fnames.append(items[0])
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

        self.set_bucket_info(num_buckets)


    def __getitem__(self, index):
        import soundfile as sf

        path_or_fp = os.path.join(self.root_dir, str(self.fnames[index]))
        _path, slice_ptr = parse_path(path_or_fp)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            path_or_fp = io.BytesIO(byte_data)
        wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")

        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, curr_sample_rate)
        return {"id": index, "source": wav}
