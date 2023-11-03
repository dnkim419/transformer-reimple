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


# ffn
class FFN(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.conv1 = nn.Conv1d(in_channels=self.config["d_model"], out_channels=self.config["d_ff"], kernel_size=1)
    self.conv2 = nn.Conv1d(in_channels=self.config["d_ff"], out_channels=self.config["d_model"], kernel_size=1)
    self.active = F.relu
    self.dropout = nn.Dropout(self.config["dropout"])

  # inputs: (batch, n_seq, d_model)
  def forward(self, inputs):
    # (batch, n_seq, d_model) -> (batch, d_model, n_seq) -> (batch, d_ff, n_seq)
    output = self.active(self.conv1(inputs.transpose(1,2)))
    # (batch, d_ff, n_seq) -> (batch, d_model, n_seq) -> (batch, n_seq, d_model)
    output = self.conv2(output).transpose(1,2)
    output = self.dropout(output)
    # output: (batch, n_seq, d_model)
    return output


# encoder
class Encoder(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    self.enc_emb = nn.Embedding(self.config["n_enc_vocab"], self.config["d_model"])
    pos_enc_table = torch.FloatTensor(get_sinusoidal(self.config["n_enc_seq"], self.config["d_model"]))
    self.pos_emb = nn.Embedding.from_pretrained(pos_enc_table, freeze=True)

    # to do: EncoderLayer
  
  # to do: forward