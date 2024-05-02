from pytorch_transformers import (WEIGHTS_NAME, BertConfig, 
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
from pytorch_transformers.modeling_bert import BertModel, BertEmbeddings, BertEncoder, BertPooler, BertLayer

from torch import nn
import os

import torch
import torch.distributed

# Model parallel version of BertForSequenceClassification
# This is the same as the original BertForSequenceClassification,
# Except it uses BertModelMP instead of BertModel.
class BertForSequenceClassificationMP(BertForSequenceClassification):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        
        self.bert = BertModelMP(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

        
# Model parallel version of BertModel, uses BertEncoderMP
# Ignores mp for the embeddings and pooler
# Because they are small and can be replicated.
class BertModelMP(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoderMPDistributed(config)
        self.pooler = BertPooler(config)
        
        self.init_weights()


# Model parallel version of BertEncoder
class BertEncoderMPDistributed(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        
        # This instance will only hold the layers assigned to this rank.
        # The layers are divided among the ranks.
        # For example, if there are 12 layers and 4 ranks,
        # rank 0 will hold layers 0, 1, 2
        # rank 1 will hold layers 3, 4, 5
        # rank 2 will hold layers 6, 7, 8
        # rank 3 will hold layers 9, 10, 11
        self.gpu_allocation = [0] * config.num_hidden_layers
        layer_count = config.num_hidden_layers // self.world_size
        for i in range(config.num_hidden_layers):
            gpu = i // layer_count
            self.gpu_allocation[i] = gpu
                
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        
        if self.rank != 0:
            torch.distributed.recv(tensor=hidden_states, src=self.rank-1)
        
        for i, layer_module in enumerate(self.layer):
            if self.gpu_allocation[i] != self.rank: 
                # If the layer is not assigned to this rank, skip it
                continue
            
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Send the hidden states to the next rank
        if self.rank == self.world_size - 1:
            torch.distributed.send(tensor=hidden_states, dst=self.rank+1)            
        else:
            # Last rank sends the hidden states back to the first rank
            torch.distributed.send(tensor=hidden_states, dst=0)
            
        # Receive the final hidden states from the last rank
        if self.rank == 0:
            torch.distributed.recv(tensor=hidden_states, src=self.world_size-1)
        
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
    