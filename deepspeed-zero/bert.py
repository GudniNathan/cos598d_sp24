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

import copy

BertArgs = None

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
        args = BertArgs
        if args.deepspeed_transformer_kernel:
            from deepspeed import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig, DeepSpeedConfig

            if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
                ds_config = DeepSpeedConfig(args.deepspeed_config)
            else:
                raise RuntimeError('deepspeed_config is not found in args.')

            cuda_config = DeepSpeedTransformerConfig(
                batch_size = ds_config.train_micro_batch_size_per_gpu,
                max_seq_length = args.max_seq_length,
                hidden_size = config.hidden_size,
                heads = config.num_attention_heads,
                attn_dropout_ratio = config.attention_probs_dropout_prob,
                hidden_dropout_ratio = config.hidden_dropout_prob,
                num_hidden_layers = config.num_hidden_layers,
                initializer_range = config.initializer_range,
                local_rank = args.local_rank if hasattr(args, 'local_rank') else -1,
                seed = args.seed,
                fp16 = ds_config.fp16_enabled,
                pre_layer_norm=True,
                attn_dropout_checkpoint=args.attention_dropout_checkpoint,
                normalize_invertible=args.normalize_invertible,
                gelu_checkpoint=args.gelu_checkpoint,
                stochastic_mode=True)

            layer = DeepSpeedTransformerLayer(cuda_config)
        else:
            layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])