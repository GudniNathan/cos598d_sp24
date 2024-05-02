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
            print(f"layer {i} to gpu {gpu}")
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
