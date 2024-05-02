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
            layer.to(gpu)
            self.gpu_allocation[i] = gpu
        self.gpu_allocation.append(gpu) # For the output layer
                
        self.layer = nn.ModuleList(layers)


    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        # Print the devices
        hidden_state_device = hidden_states.device
        print(f"Hidden states device: {hidden_state_device}")
        attention_mask_device = attention_mask.device
        print(f"Attention mask device: {attention_mask_device}")
        if head_mask is not None:
            head_mask_device = head_mask.device
            print(f"Head mask device: {head_mask_device}")
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0].to(self.gpu_allocation[i+1])

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

            

        