import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# sinusoidal position representations
def get_sinusoidal(n_seq, d_model):
  '''
  Args:
      n_seq: sequence 길이 (=한 문장 내 토큰 개수)
      d_model: (=512)
  '''
  def cal_angle(i_seq, i_dmodel):
    return i_seq / np.power(10000, 2 * (i_dmodel // 2) / d_model)

  def get_pos_enc(i_seq):
    return [cal_angle(i_seq, i_dmodel) for i_dmodel in range(d_model)]

  pos_enc_table = np.array([get_pos_enc(i_seq) for i_seq in range(n_seq)])
  pos_enc_table[:, 0::2] = np.sin(pos_enc_table[:, 0::2]) # even idx
  pos_enc_table[:, 1::2] = np.cos(pos_enc_table[:, 1::2]) # odd idx

  return pos_enc_table


# Feed-Forward Network
class FFN(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.conv1 = nn.Conv1d(in_channels=self.config["d_model"], out_channels=self.config["d_ff"], kernel_size=1)
    self.conv2 = nn.Conv1d(in_channels=self.config["d_ff"], out_channels=self.config["d_model"], kernel_size=1)
    self.active = F.relu
    self.dropout = nn.Dropout(self.config["dropout"])

  # inputs: (batch_size, n_seq, d_model)
  def forward(self, inputs):
    # (batch_size, n_seq, d_model) -> (batch_size, d_model, n_seq) -> (batch_size, d_ff, n_seq)
    output = self.active(self.conv1(inputs.transpose(1,2)))
    # (batch_size, d_ff, n_seq) -> (batch_size, d_model, n_seq) -> (batch_size, n_seq, d_model)
    output = self.conv2(output).transpose(1,2)
    output = self.dropout(output)
    # output: (batch_size, n_seq, d_model)
    return output


# attention pad mask
def get_attn_pad_mask(query, key, i_pad):
  '''
  Args:
      query: query(Q) (batch_size, 문장 내 토큰 개수)
      key: key(K) (batch_size, 문장 내 토큰 개수)
      * 전처리 했으므로 배치 내 토큰 개수 동일
      i_pad: padding 인덱스 (=0)
  '''
  batch_size, len_q = query.size()
  batch_size, len_k = key.size()
  # (batch_size, len_q, len_k)
  mask = key.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)
  return mask


# attention decoder mask
def get_attn_decoder_mask(seq):
  '''
  Args:
      seq: (batch_size, 문장 내 토큰 개수)
  '''
  mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
  # (batch_size, len_seq, len_seq)
  mask = mask.triu(diagonal=1)
  return mask


# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.dropout = nn.Dropout(self.config["dropout"])
    self.scale = 1 / (self.config["d_h"] ** 0.5)

  def forward(self, Q, K, V, attn_mask):
    '''
    Args:
        Q: (batch_size, h, len_q, d_h)
        K: (batch_size, h, len_k, d_h)
        V: (batch_size, h, len_v, d_h)
        attn_mask: (batch_size, h, len_q, len_k)
    '''
    # (batch_size, h, len_q, len_k)
    affinities = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
    affinities.masked_fill_(attn_mask, -1e9)
    # (batch_size, h, len_q, len_k)
    attn_weights = nn.Softmax(dim=-1)(affinities)
    attn_weights = self.dropout(attn_weights)
    # (batch_size, h, len_q, d_h)
    output = torch.matmul(attn_weights, V)
    # (batch_size, h, len_q, d_h), (batch_size, h, len_q, len_k)
    return output, attn_weights


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.W_Q = nn.Linear(self.config['d_model'], self.config['h'] * self.config['d_h'])
    self.W_K = nn.Linear(self.config['d_model'], self.config['h'] * self.config['d_h'])
    self.W_V = nn.Linear(self.config['d_model'], self.config['h'] * self.config['d_h'])
    self.scaled_dot_attn = ScaledDotProductAttention(self.config)
    self.linear = nn.Linear(self.config['h'] * self.config['d_h'], self.config['d_model'])
    self.dropout = nn.Dropout(self.config['dropout'])

  def forward(self, Q, K, V, attn_mask):
    '''
    Args:
        Q: (batch_size, len_q, d_model)
        K: (batch_size, len_q, d_model)
        V: (batch_size, len_q, d_model)
        attn_mask: (batch_size, len_q, len_k)
    '''
    # linearly project the queries, keys and values
    # (batch_size, len_q, d_model) * (d_model, h * d_h) = (batch_size, len_q, h * d_h)
    # -> (batch_size, len_q, h, d_h)
    # -> (batch_size, h, len_q, d_h)
    pjted_Q = self.W_Q(Q).view(self.config['batch_size'], -1, self.config['h'], self.config['d_h']).transpose(1,2)
    pjted_K = self.W_K(K).view(self.config['batch_size'], -1, self.config['h'], self.config['d_h']).transpose(1,2)
    pjted_V = self.W_V(V).view(self.config['batch_size'], -1, self.config['h'], self.config['d_h']).transpose(1,2)
    # (batch_size, len_q, len_k) -> (batch_size, h, len_q, len_k)
    attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config['h'], 1, 1)
    # scaled dot product attention
    # (batch_size, h, len_q, d_h), (batch_size, h, len_q, len_k)
    context, attn_weights = self.scaled_dot_attn(pjted_Q, pjted_K, pjted_V, attn_mask)
    # concat
    # (batch_size, h, len_q, d_h) -> (batch_size, len_q, h * d_h)
    context= context.transpose(1, 2).contiguous().view(self.config['batch_size'], -1, self.config['h'] * self.config['d_h'])
    # linear
    # (batch_size, len_q, h * d_h) * (h * d_h, d_model)
    # -> (batch_size, len_q, d_model)
    output = self.linear(context)
    output = self.dropout(output)
    # (batch_size, len_q, d_model), (batch_size, h, len_q, len_k)
    return output, attn_weights


# encoder layer
class EncoderLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.self_attn = MultiHeadAttention(self.config)
    self.layer_norm1 = nn.LayerNorm(self.config["d_model"], eps = self.config["layer_norm_epsilon"])
    self.ffn = FFN(self.config)
    self.layer_norm2 = nn.LayerNorm(self.config["d_model"], eps = self.config["layer_norm_epsilon"])

  '''
  Args:
      inputs: (batch_size, len_seq, d_model)
      attn_mask: (batch_size, len_q, len_k)
  '''
  def forward(self, inputs, attn_mask):
    # (batch_size, len_q, d_model), (batch_size, h, len_q, len_k)
    attn_output, attn_weights = self.self_attn(inputs, inputs, inputs, attn_mask)
    # (batch_size, len_q, d_model)
    attn_output = self.layer_norm1(inputs + attn_output)
    # (batch_size, len_q, d_model)
    ffn_output = self.ffn(attn_output)
    # (batch_size, len_q, d_model)
    ffn_output = self.layer_norm2(ffn_output + attn_output)
    # (batch_size, len_q, d_model), (batch_size, h, len_q, len_k)
    return ffn_output, attn_weights


# Encoder
class Encoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.enc_emb = nn.Embedding(self.config["n_enc_vocab"], self.config["d_model"])
    pos_enc_table = torch.FloatTensor(get_sinusoidal(self.config["n_enc_seq"] + 1, self.config["d_model"]))
    self.pos_emb = nn.Embedding.from_pretrained(pos_enc_table, freeze=True)

    self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config["n_layer"])])

  '''
  Args
      inputs: (batch_size, len_seq)
  '''
  def forward(self, inputs):
    # (batch_size, len_enc_seq)
    positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
    pos_mask = inputs.eq(self.config["i_pad"])
    positions.masked_fill_(pos_mask, 0)

    # (batch_size, len_enc_seq, d_model)
    output = self.enc_emb(inputs) + self.pos_emb(positions)

    # (batch_size, len_enc_seq, len_enc_seq)
    attn_mask = get_attn_pad_mask(inputs, inputs, self.config["i_pad"])

    attn_weights_history = list([])
    for layer in self.layers:
      # (batch_size, len_enc_seq, d_model), (batch_size, h, len_enc_seq, len_enc_seq)
      output, attn_weights = layer(output, attn_mask)
      attn_weights_history.append(attn_weights)

    # (batch_size, len_enc_seq, d_model), [(batch_size, h, len_enc_seq, len_enc_seq)]
    return output, attn_weights_history
  

# decoder layer
class DecoderLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.self_attn = MultiHeadAttention(self.config)
    self.layer_norm1 = nn.LayerNorm(self.config["d_model"], eps = self.config["layer_norm_epsilon"])
    self.enc_dec_attn = MultiHeadAttention(self.config)
    self.layer_norm2 = nn.LayerNorm(self.config["d_model"], eps = self.config["layer_norm_epsilon"])
    self.ffn = FFN(self.config)
    self.layer_norm3 = nn.LayerNorm(self.config["d_model"], eps = self.config["layer_norm_epsilon"])

  '''
  Args:
      dec_inputs: (batch_size, len_seq, d_model)
      enc_outputs: (batch_size, len_enc_seq, d_model)
      self_attn_mask: (batch_size, len_dec_seq, len_dec_seq)
      enc_dec_attn_mask: (batch_size, len_dec_seq, len_enc_seq)
  '''
  def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_dec_attn_mask):
    # (batch_size, len_dec_seq, d_model), (batch_size, h, len_dec_seq, len_dec_seq)
    self_attn_output, self_attn_weights = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
    self_attn_output = self.layer_norm1(dec_inputs + self_attn_output)
    # (batch_size, len_dec_seq, d_model), (batch_size, h, len_dec_seq, len_ebc_seq)
    enc_dec_attn_output, enc_dec_attn_weights = self.enc_dec_attn(self_attn_output, enc_outputs, enc_outputs, enc_dec_attn_mask)
    enc_dec_attn_output = self.layer_norm2(self_attn_output + enc_dec_attn_output)
    # (batch_size, len_dec_seq, d_model)
    ffn_output = self.ffn(enc_dec_attn_output)
    ffn_output = self.layer_norm3(enc_dec_attn_output + ffn_output)
    # (batch_size, len_dec_seq, d_model), (batch_size, h, len_dec_seq, len_dec_seq), (batch_size, h, len_dec_seq, len_ebc_seq)
    return ffn_output, self_attn_weights, enc_dec_attn_weights  
  

# Decoder 
class Decoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.dec_emb = nn.Embedding(self.config["n_dec_vocab"], self.config["d_model"])
    pos_enc_table = torch.FloatTensor(get_sinusoidal(self.config["n_dec_seq"] + 1, self.config["d_model"]))
    self.pos_emb = nn.Embedding.from_pretrained(pos_enc_table, freeze=True)

    self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config["n_layer"])])

  '''
  Args:
      dec_inputs: (batch_size, len_dec_seq, d_model)
      enc_inputs: (batch_size, len_enc_seq, d_model)
      enc_outputs: (batch_size, len_enc_seq, d_model)
  '''
  def forward(self, dec_inputs, enc_inputs, enc_outputs):
    # (batch_size, len_enc_seq)
    positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device, dtype=dec_inputs.dtype).expand(dec_inputs.size(0), dec_inputs.size(1)).contiguous() + 1
    pos_mask = dec_inputs.eq(self.config["i_pad"])
    positions.masked_fill_(pos_mask, 0)

    # (batch_size, n_dec_seq, d_model)
    dec_output = self.dec_emb(dec_inputs) + self.pos_emb(positions)

    # (batch_size, len_dec_seq, len_dec_seq)
    attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config["i_pad"])
    # (batch_size, len_dec_seq, len_dec_seq)
    attn_decoder_mask = get_attn_decoder_mask(dec_inputs)
    # (batch_size, len_dec_seq, len_dec_seq)
    self_attn_mask = torch.gt((attn_pad_mask + attn_decoder_mask), 0)
    # (batch_size, len_dec_seq, len_enc_seq)
    enc_dec_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config["i_pad"])

    self_attn_weights_history, enc_dec_attn_weights_history = list([]), list([])
    for layer in self.layers:
      # (batch_size, len_dec_seq, d_model), (batch_size, h, len_dec_seq, len_dec_seq), (batch_size, h, len_dec_seq, len_ebc_seq)
      output, self_attn_weights, enc_dec_attn_weights = layer(dec_output, enc_outputs, self_attn_mask, enc_dec_attn_mask)
      self_attn_weights_history.append(self_attn_weights)
      enc_dec_attn_weights_history.append(enc_dec_attn_weights)
    # (batch_size, len_dec_seq, d_model), (batch_size, h, len_dec_seq, len_dec_seq), (batch_size, h, len_dec_seq, len_ebc_seq)
    return output, self_attn_weights_history, enc_dec_attn_weights_history