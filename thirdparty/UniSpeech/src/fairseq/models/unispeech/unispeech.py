# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from omegaconf import II
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import BaseFairseqModel, register_model
from fairseq.tasks.audio_pretraining import AudioPretrainingTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config, Wav2Vec2Model
from fairseq.models.wav2vec.wav2vec2_asr import Linear


@dataclass
class UniSpeechConfig(Wav2Vec2Config):
    replace_prob: float = field(
        default=0.5, metadata={"help": "replacement probability for CTC pre-training"}
    )
    final_dropout: float = field(
        default=0.1
    )

@register_model("unispeech", dataclass=UniSpeechConfig)
class Unispeech(BaseFairseqModel):
    def __init__(self, cfg, w2v_encoder):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.cfg = cfg

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls,  cfg: UniSpeechConfig, task: AudioPretrainingTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoder(cfg, task)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x

    def remove_pretraining_modules(self):
        self.w2v_encoder.proj = None

class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, cfg, task):
        super().__init__(task.source_dictionary)
        self.w2v_model = Wav2Vec2Model(cfg) 

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.num_updates = 0
        self.replace_prob = cfg.replace_prob

        d = cfg.encoder_embed_dim

        if task.target_dictionary is not None:
            self.proj = Linear(d, len(task.target_dictionary))
        elif getattr(cfg, 'decoder_embed_dim', d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.training,
        }

        contrastive_res = self.w2v_model(**w2v_args)
        x = contrastive_res["features"]
        padding_mask = contrastive_res["padding_mask"]
        q = contrastive_res["q"]

        if tbc:
            x = x.transpose(0, 1)
            q = q.transpose(0, 1)
        replace_mat = torch.empty(x.size(0), x.size(1)).fill_(self.replace_prob)
        replace_mat = torch.bernoulli(replace_mat).bool().to(x.device)
        replace_mat = replace_mat.unsqueeze(-1)
        x = x.masked_fill(replace_mat, 0.0) + q.masked_fill(~replace_mat, 0.0)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "contrastive_res": contrastive_res,
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict



