from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialize the linear transformation layers for key, value, query
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # this dropout is applied to normalized attention scores following the original implementation of transformer
        # although it is a bit unusual, we empirically observe that it yields better performance
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # next, we need to produce multiple heads for the proj
        # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
        proj = proj.view(bs, seq_len, self.num_attention_heads,
                         self.attention_head_size)
        # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
        # attention scores are calculated by multiply query and key
        # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
        # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
        # before normalizing the scores, use the attention mask to mask out the padding token scores
        # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number

        # normalize the scores
        # multiply the attention scores to the value and get back V'
        # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]

        # TODO
        # raise NotImplementedError

        (bs, num_heads, seq_len, head_size) = key.shape

        attention_scores = (query.matmul(
            key.transpose(2, 3))) / math.sqrt(head_size)
        attention_scores = attention_scores + attention_mask

        # Normalize the scores
        attention_scores = F.softmax(attention_scores, dim=-1)
        scores = self.dropout(attention_scores)

        # Multiply the attention scores to the value and get back V'
        res = scores.matmul(value)

        # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
        out = (res.transpose(1, 2)).reshape(bs, seq_len, -1)
        return out

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bs, seq_len, hidden_state]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_state]
        """
        # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
        # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        # calculate the multi-head attention
        attn_value = self.attention(
            key_layer, query_layer, value_layer, attention_mask)
        return attn_value


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # multi-head attention
        self.self_attention = BertSelfAttention(config)
        # add-norm
        self.attention_dense = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # feed forward
        self.interm_dense = nn.Linear(
            config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # another add-norm
        self.out_dense = nn.Linear(
            config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        """
        this function is applied after the multi-head attention layer or the feed forward layer
        input: the input of the previous layer
        output: the output of the previous layer
        dense_layer: used to transform the output
        dropout: the dropout to be applied 
        ln_layer: the layer norm to be applied
        """
        # Hint: Remember that BERT applies to the output of each sub-layer, before it is added to the sub-layer input and normalized
        # TODO
        # raise NotImplementedError
        output = dense_layer(output)
        output = dropout(output)
        output = ln_layer(output + input)

        return output

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
        as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
        each block consists of 
        1. a multi-head attention layer (BertSelfAttention)
        2. a add-norm that takes the input and output of the multi-head attention layer
        3. a feed forward layer
        4. a add-norm that takes the input and output of the feed forward layer
        """
        # TODO
        # raise NotImplementedError

        # Multi-head self-attention
        attention_output = self.self_attention(hidden_states, attention_mask)

        # Add-Norm after attention
        attention_output = self.add_norm(hidden_states,
                                         attention_output,
                                         self.attention_dense,
                                         self.attention_dropout,
                                         self.attention_layer_norm)

        # Feed-forward network
        intermediate_output = self.interm_dense(attention_output)
        intermediate_output = self.interm_af(intermediate_output)

        # Add-Norm after Feed-forward
        layer_output = self.add_norm(attention_output,
                                     intermediate_output,
                                     self.out_dense,
                                     self.out_dropout,
                                     self.out_layer_norm)

        return layer_output


class BertModel(BertPreTrainedModel):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # embedding
        self.word_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.tk_type_embedding = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        self.embed_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is a constant, register to buffer
        position_ids = torch.arange(
            config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # bert encoder
        self.bert_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)])

        # for [CLS] token
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        # Get word embedding from self.word_embedding into input_embeds.
        inputs_embeds = None
        # TODO
        # raise NotImplementedError
        inputs_embeds = self.word_embedding(input_ids)

        # Get position index and position embedding from self.pos_embedding into pos_embeds.
        pos_ids = self.position_ids[:, :seq_length]

        pos_embeds = None
        # TODO
        # raise NotImplementedError
        pos_embeds = self.pos_embedding(pos_ids)[:, :inputs_embeds.size(1), :]
        pos_embeds = pos_embeds.expand_as(inputs_embeds)

        # Get token type ids, since we are not consider token type, just a placeholder.
        tk_type_ids = torch.zeros(
            input_shape, dtype=torch.long, device=input_ids.device)
        tk_type_embeds = self.tk_type_embedding(tk_type_ids)

        # Add three embeddings together; then apply embed_layer_norm and dropout and return.
        # TODO
        # raise NotImplementedError

        embeds = inputs_embeds + pos_embeds + tk_type_embeds
        embeds = self.embed_layer_norm(embeds)
        embeds = self.embed_dropout(embeds)

        return embeds

    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # get the extended attention mask for self attention
        # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
        # non-padding tokens with 0 and padding tokens with a large negative number
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(
            attention_mask, self.dtype)

        # pass the hidden states through the encoder layers
        for i, layer_module in enumerate(self.bert_layers):
            # feed the encoding from the last bert_layer to the next
            hidden_states = layer_module(
                hidden_states, extended_attention_mask)

        return hidden_states

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # get the embedding for each input token
        embedding_output = self.embed(input_ids=input_ids)

        # feed to a transformer (a stack of BertLayers)
        sequence_output = self.encode(
            embedding_output, attention_mask=attention_mask)

        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        first_tk = self.pooler_dense(first_tk)
        first_tk = self.pooler_af(first_tk)

        return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}

#### RoPE ####


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        # pe.requires_grad = False
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.max_len = max_len
        self.d_model = d_model
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, max_len)
        batch = x.size(0)

        result = self.pe.repeat(batch, 1, 1).to(x.device)

        # x = x + self.pe[:, :x.size(1)]
        # self.pe[:, :x.size(1)]
        # print(f"first: {self.pe.shape}")
        # print(f"third: {self.pe.repeat(batch, 1, 1).shape}")
        return result  # (batch, max_len, d_model)


class RoBertModel(BertModel):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """

    def __init__(self, config):
        super(RoBertModel, self).__init__(config)
        self.config = config

        # self.pos_embedding = nn.Embedding(
        #     config.max_position_embeddings, config.hidden_size)

        self.pos_embedding = PositionalEncoding(
            config.max_position_embeddings, config.hidden_size)

#### End of RoPE ####


#### RMSNorm ####

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Calculate the root mean square
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize the input
        x_norm = x / rms
        return self.weight * x_norm


class RMSRoBertLayer(BertLayer):
    def __init__(self, config):
        super(RMSRoBertLayer, self).__init__(config)

        self.attention_layer_norm = RMSNorm(
            config.hidden_size)

        self.out_layer_norm = RMSNorm(
            config.hidden_size)


class RMSRoBertModel(BertModel):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """

    def __init__(self, config):
        super(RMSRoBertModel, self).__init__(config)
        self.config = config

        self.embed_layer_norm = RMSNorm(
            config.hidden_size)

        self.bert_layers = nn.ModuleList(
            [RMSRoBertLayer(config) for _ in range(config.num_hidden_layers)])

        self.pos_embedding = PositionalEncoding(
            config.max_position_embeddings, config.hidden_size)

#### End of RMS ####

#### SwiGLU ####


class SwiGLU(nn.Module):
    def __init__(self, hidden_size):
        super(SwiGLU, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, x):
        x, gate = self.linear(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class SRMSRoBertLayer(BertLayer):
    def __init__(self, config):
        super(SRMSRoBertLayer, self).__init__(config)

        self.attention_layer_norm = RMSNorm(
            config.hidden_size)

        self.out_layer_norm = RMSNorm(
            config.hidden_size)

        self.interm_af = SwiGLU


class SRMSRoBertModel(BertModel):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """

    def __init__(self, config):
        super(SRMSRoBertModel, self).__init__(config)
        self.config = config

        self.embed_layer_norm = RMSNorm(
            config.hidden_size)

        self.bert_layers = nn.ModuleList(
            [SRMSRoBertLayer(config) for _ in range(config.num_hidden_layers)])

        self.pos_embedding = PositionalEncoding(
            config.max_position_embeddings, config.hidden_size)

#### End of SwiGLU ####
