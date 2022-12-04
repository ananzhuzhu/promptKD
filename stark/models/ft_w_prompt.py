# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead
from modules.modeling_sparse_bert import SparseBertModel

import collections


class FTPrompt(BertPreTrainedModel):
    Output = collections.namedtuple(
        "Output", 
        (
            "logit",
            "prediction", 
            "label",
        )
    )
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = SparseBertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def forward(self, inputs, head_mask=None, neuron_mask=None):
        text_indices, text_mask, text_segments, mask_position, verbalizer_indices, verbalizer_mask, label = inputs
        # text_indices, text_mask, text_segments, label = inputs

        # Gather knowledge.
        if neuron_mask is None:
            hidden_states = \
                self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments)[0]
        else:
            hidden_states = \
                self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments, head_mask=head_mask, neuron_mask=neuron_mask)[0]

        # Mapping.
        # There is no mapping for KD.
        # Logit.
        hidden_states = torch.gather(hidden_states, 1, mask_position.unsqueeze(2).expand(-1, -1, hidden_states.shape[2])).squeeze(1)
        logit = self.cls(hidden_states)
        logit = torch.gather(logit.unsqueeze(1).expand(-1, verbalizer_indices.shape[1], -1), 2, verbalizer_indices)
        logit = torch.sum(logit * verbalizer_mask.float(), 2) / verbalizer_mask.float().sum(2)
        
        # Mask and reshape.
        # There is no mask or reshape for KD.

        prediction = logit.argmax(-1)

        return FTPrompt.Output(
            logit=logit,
            prediction=prediction, 
            label=label,
        )

    @staticmethod
    def loss_fn(output):
        loss = F.cross_entropy(output.logit, output.label, reduction="mean")
        return loss

        
