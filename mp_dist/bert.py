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
        super().__init__(config)
        self.bert = BertModelMP(config)
        self.dropout = self.dropout.to(0)
        self.classifier = self.classifier.to(0)

        
# Model parallel version of BertModel, uses BertEncoderMP
# Ignores mp for the embeddings and pooler
# Because they are small and can be replicated.
class BertModelMP(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config).to(0)
        self.encoder = BertEncoderMP(config)
        self.pooler = BertPooler(config).to(0)

# Model parallel version of BertEncoder
class BertEncoderMP(BertEncoder):
    # Need to split the layers among the GPUs
    # Gpu 0 gets layers 0 to 5, gpu 1 gets 6 to 11, etc.
    def __init__(self, config):
        super().__init__(config)
        layers = [BertLayer(config) for _ in range(config.num_hidden_layers)]
        gpu_count = torch.cuda.device_count()
        layer_count = config.num_hidden_layers // gpu_count
        
        self.gpu_allocation = [0] * config.num_hidden_layers
        for i, layer in enumerate(self.layer):
            gpu = i // layer_count
            layers[i] = layer.to(f"cuda:{gpu}")
            self.gpu_allocation[i] = gpu
        self.gpu_allocation.append(gpu) # For the output layer
                
        self.layer = nn.ModuleList(layers)


    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        for i, layer_module in enumerate(self.layer):
            if i > 0 and self.gpu_allocation[i] != self.gpu_allocation[i-1]:
                hidden_states = hidden_states.to(self.gpu_allocation[i])
                attention_mask = attention_mask.to(self.gpu_allocation[i])
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0].to(self.gpu_allocation[i+1])

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = hidden_states.to(0)
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
            
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

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
        layer_count = config.num_hidden_layers // world_size
        layers = []
        for i in range(config.num_hidden_layers):
            gpu = (i+1) // layer_count
            self.gpu_allocation[i] = gpu
            if gpu == rank:
                layers.append(BertLayer(config))
                
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        
        if self.rank != 0:
            torch.distributed.recv(tensor=hidden_states, src=self.rank-1)
        
        for i, layer_module in enumerate(self.layer):
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
    