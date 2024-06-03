# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.data.dictionary import Dictionary
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.hubert import HubertConfig, HubertModel
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask,
)

logger = logging.getLogger(__name__)


@dataclass
class ILSHubertConfig(HubertConfig):
    #relative position embedding
    relative_position_embedding: bool = field(
        default=False,
        metadata={"help": "whether to use the relative position embedding, (bucket relpos embedding by default)"}
    )
    num_buckets: int = field(
        default=320,
        metadata={"help": "the number of buckets for relative position embedding"}
    )
    max_distance: int = field(
        default=800,
        metadata={"help": "the maximum distance for computing relative bias, beyond which will assign the same embedding"}
    )

    # ILS-SSL params
    weighted_sum: bool = field(
        default=False
    )
    predict_layers: str = field(
        default="[12]"
    )
    separate_label_embeds: bool = field(
        default=False
    )
    separate_layer_targets: bool = field(
        default=False
    )


@register_model("ils_hubert", dataclass=ILSHubertConfig)
class ILSHubertModel(HubertModel):
    def __init__(
        self,
        cfg: ILSHubertConfig,
        task_cfg: HubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__(cfg, task_cfg, dictionaries)
        logger.info(f"HubertModel Config: {cfg}")

        self.predict_layers = eval(cfg.predict_layers)
        self.separate_label_embeds = cfg.separate_label_embeds
        self.separate_layer_targets = cfg.separate_layer_targets
        self.weighted_sum = cfg.weighted_sum

        self.layer_norm_first = cfg.layer_norm_first
        if self.layer_norm_first:
            self.post_layer_norm = torch.nn.Sequential(*[LayerNorm(cfg.encoder_embed_dim) for _ in range(len(self.predict_layers))])

        if self.separate_label_embeds:
            if self.separate_layer_targets or not self.untie_final_proj:
                self.final_proj = torch.nn.Sequential(*[nn.Linear(
                   cfg.encoder_embed_dim, cfg.final_dim)
                    for _ in range(len(self.predict_layers))])
            else:
                self.final_proj = torch.nn.Sequential(*[nn.Linear(
                    cfg.encoder_embed_dim, cfg.final_dim * len(dictionaries))
                    for _ in range(len(self.predict_layers))])
        else:
            if self.separate_layer_targets or not self.untie_final_proj:
                self.final_proj = nn.Linear(cfg.encoder_embed_dim, cfg.final_dim)
            else:
                self.final_proj = nn.Linear(cfg.encoder_embed_dim, cfg.final_dim * len(dictionaries))

        if self.weighted_sum:
            self.weights = nn.Parameter(torch.zeros(len(self.predict_layers)))
        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info(
                "cannot find dictionary. assume will be used for fine-tuning"
            )
        else:
            self.num_classes = [len(d) for d in dictionaries]
            layer_dim = len(self.predict_layers) if self.separate_layer_targets or self.separate_label_embeds else 1
            embed_dim = sum(self.num_classes) if not self.separate_layer_targets else max(self.num_classes)
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(layer_dim, embed_dim, cfg.final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

    @classmethod
    def build_model(cls, cfg: ILSHubertConfig, task: HubertPretrainingTask):
        """Build a new model instance."""

        model = ILSHubertModel(cfg, task.cfg, task.dictionaries)
        return model

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)


        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)


        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(
                features, padding_mask, target_list
            )
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=self.predict_layers
        )

        result = {"x": x, "padding_mask": padding_mask, "features": features, "layer_results": layer_results}

        if features_only:
            if self.layer_norm_first and output_layer is not None:
                result['x'] = self.post_layer_norm[-1](x)
            return result

        layer_results = [layer_x.transpose(0, 1) for i, (layer_x, _) in enumerate(layer_results)]

        if not (x == layer_results[-1]).all():
            print("{} {} {} {}".format((x == layer_results[-1]).shape, (x == layer_results[-1]).float().sum(),
                (x - layer_results[-1]).float().sum(), (x - layer_results[-1]).float().abs().max(),))

        if self.layer_norm_first:
            layer_results = [layernorm(x) for x, layernorm in zip(layer_results, self.post_layer_norm)]

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)


        logit_m_list = []
        logit_u_list = []
        proj_x_m_list = []
        proj_x_u_list = []

        if self.separate_layer_targets:
            assert len(layer_results) == len(self.final_proj)
            assert len(layer_results) == len(self.label_embs_concat)

        for i, layer_x in enumerate(layer_results):  #, final_proj, label_embs in zip(layer_results, self.final_proj, label_embs_concat):
            if self.separate_label_embeds:
                final_proj = self.final_proj[i]
            else:
                final_proj = self.final_proj

            if self.separate_label_embeds or self.separate_layer_targets:
                label_embs = self.label_embs_concat[i]
            else:
                label_embs = self.label_embs_concat[0]

            if not self.separate_layer_targets:
                label_embs_list = label_embs.split(self.num_classes, 0)
            else:
                label_embs_list = [label_embs[:self.num_classes[i]]]

            if not self.skip_masked:
                masked_indices = torch.logical_and(~padding_mask, mask_indices)
                proj_x_m = final_proj(layer_x[masked_indices])

                if self.separate_layer_targets:
                    proj_x_m_list = [proj_x_m]
                    logit_m_list += [
                        compute_pred(proj_x_m, target_list[i][masked_indices], label_embs_list[0])
                    ]
                else:
                    if self.untie_final_proj:
                        proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
                    else:
                        proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
                    logit_m_list += [
                        compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                        for i, (proj_x_m, t) in enumerate(
                            zip(proj_x_m_list, target_list)
                        )
                    ]
            else:
                logit_m_list += [None for _ in target_list]

            if not self.skip_nomask:
                nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
                proj_x_u = final_proj(layer_x[nomask_indices])
                if self.separate_layer_targets:
                    proj_x_u_list = [proj_x_u]
                    logit_u_list += [
                        compute_pred(proj_x_u, target_list[i][nomask_indices], label_embs_list[0])
                    ]
                else:
                    if self.untie_final_proj:
                        proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
                    else:
                        proj_x_u_list = [proj_x_u for _ in range(len(target_list))]
                    logit_u_list += [
                        compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                        for i, (proj_x_u, t) in enumerate(
                            zip(proj_x_u_list, target_list)
                        )
                    ]
            else:
                logit_u_list += [None for _ in target_list]

        result["logit_m_list"] = logit_m_list
        result["logit_u_list"] = logit_u_list
        result["padding_mask"] = padding_mask
        result["features_pen"] = features_pen
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        ret_layer_results: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        if ret_layer_results:
            return (feature, res["layer_results"]), res["padding_mask"]
        return feature, res["padding_mask"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [
            x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list
        ]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
        self.label_embs_concat = None
