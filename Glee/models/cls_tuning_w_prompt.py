# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel


Output = collections.namedtuple(
    "Output", 
    (
        'loss', 
        'prediction', 
        'label',
    )
)


class CLSTuningWPrompt(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        if config.activation == "relu":
            self.cls = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_size, config.num_labels),
            )
        else:
            self.cls = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_size, config.num_labels),
            )
        self.init_weights()

    def forward(self, inputs):
        text_indices, text_mask, text_segments, mask_position, verbalizer_indices, verbalizer_mask, label = inputs

        hidden_states = self.bert(text_indices, attention_mask=text_mask, token_type_ids=text_segments)[0]
        hidden_states = torch.gather(hidden_states, 1, mask_position.unsqueeze(2).expand(-1, -1, hidden_states.shape[2])).squeeze(1)
        
        logit = self.cls(hidden_states)
        
        if logit.shape[-1] == 1:
            loss = F.mse_loss(logit.squeeze(-1), label.float(), reduction='none')
            prediction = logit.squeeze(-1)
            label = label.float()
        else:
            loss = F.cross_entropy(logit, label, reduction='none')
            prediction = logit.argmax(-1)
        return Output(
            loss=loss, 
            prediction=prediction, 
            label=label,
        )


        
