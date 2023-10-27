import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sinusoidal position representations
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