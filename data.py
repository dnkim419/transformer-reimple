# data.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# mt Dataset
class MtDataset(Dataset):
  def __init__(self, src_vocab, trg_vocab, df, src_name, trg_name):
    self.src_vocab  = src_vocab
    self.trg_vocab = trg_vocab
    self.src_train = []
    self.trg_train = []

    for idx, row in df.iterrows():
      src_line = row[src_name]
      trg_line = row[trg_name]
      if type(src_line) != str or type(trg_line) != str:
        continue
      # src 문장, trg 문장 각각 tokenize
      self.src_train.append(src_vocab.encode_as_ids(src_line))
      self.trg_train.append(trg_vocab.encode_as_ids(trg_line))

  def __len__(self):
    assert len(self.src_train) == len(self.trg_train)
    return len(self.src_train)

  def __getitem__(self, idx):
    return (torch.tensor(self.src_train[idx]), torch.tensor(self.trg_train[idx]))


# mt data collate_fn
# 배치 단위로 데이터 처리
def mt_collate_fn(inputs):
  enc_inputs, dec_inputs = list(zip(*inputs)) # to do

  # 입력 길이가 다르므로 입력 최대 길이에 맟춰 padding(0) 추가
  enc_inputs = torch.nn.utils.rnn.pad_sequence(enc_inputs, batch_first=True)
  dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True)

  batch = [
      enc_inputs,
      dec_inputs
  ]

  return batch # DataLoader iterate 할 때 return됨


# DataLoader
def build_mt_data_loader(src_vocab, trg_vocab, df, src_name, trg_name, args, shuffle=True):
  # Dataset 생성
  dataset = MtDataset(src_vocab, trg_vocab, df, src_name, trg_name)
  if 1 < args['n_gpu'] and shuffle:
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=args['batch'], sampler=sampler, collate_fn=mt_collate_fn)
  else:
    sampler = None
    loader = DataLoader(dataset, batch_size=args['batch'], sampler=sampler, shuffle=shuffle, collate_fn=mt_collate_fn)

  return loader, sampler