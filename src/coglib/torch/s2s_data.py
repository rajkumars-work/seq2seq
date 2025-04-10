from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..utils.prep import Prep

# from .datasets import CogsDS
from .datasets import CogsDDS


class SData:
  BATCH_SIZE = 32
  VAL_ROWS = 1000
  UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
  BOS, EOS = "[START]", "[END]"

  def __init__(self, name: str, data: str, uri_prep=True, batch_first=False):
    self.name = name
    self.data = data
    self.batch_first = batch_first
    self.pr = Prep(name, data, uri_prep=uri_prep)
    self.ct = self.pr.ct
    self.tt = self.pr.tt

  def get_vocab_sizes(self):
    return self.ct.vocab_size(), self.tt.vocab_size()

  # returns ([s_src_len, batch_size], [t_src_len, batch_size])
  def to_tensor(self, token_ids: List[int]):
    return torch.tensor(token_ids, dtype=torch.int)

  # function to add BOS/EOS and create tensor for input sequence indices and pad
  # the padding transposes the output to [max_length, batch_size] (batch_first=False)
  def collate_fn(self, batch):
    sbatch = [self.to_tensor(self.ct.tokenize(s)) for s, _ in batch]
    tbatch = [self.to_tensor(self.tt.tokenize(t)) for _, t in batch]
    sbatch = pad_sequence(sbatch, self.batch_first, padding_value=SData.PAD_IDX)
    tbatch = pad_sequence(tbatch, self.batch_first, padding_value=SData.PAD_IDX)
    return sbatch, tbatch

  # datasets and dataloaders
  def get_dls(self, dist: bool = False):
    train_ds = CogsDDS(self.name, SData.VAL_ROWS)
    train_sampler = DistributedSampler(train_ds, shuffle=False) if dist else None
    train_dl = DataLoader(
      train_ds,
      batch_size=SData.BATCH_SIZE,
      collate_fn=self.collate_fn,
      sampler=train_sampler,
    )

    val_ds = CogsDDS(self.name, 0, SData.VAL_ROWS)
    val_sampler = DistributedSampler(val_ds, shuffle=False) if dist else None
    val_dl = DataLoader(
      val_ds,
      batch_size=SData.BATCH_SIZE,
      collate_fn=self.collate_fn,
      sampler=val_sampler,
    )

    return train_dl, val_dl

  def get_transforms(self):
    def st(s):
      return self.to_tensor(self.ct.tokenize(s, mark=False))

    return st, self.pr.tt.vocab, self.pr.encode_context

  def get_dss(self):
    train_ds = CogsDDS(self.name, SData.VAL_ROWS)
    val_ds = CogsDDS(self.name, 0, SData.VAL_ROWS)
    return train_ds, val_ds

  def get_pp_func(self):
    def post_process(r: str) -> str:
      return self.dp.decode_targets(r)

    return post_process

  def encode_context(self, cs: str) -> str:
    return " ".join(self.cm.encode_as_pieces(cs))


if __name__ == "__main__":
  name = "por_en_torch"
  data = "gs://cogs-data/datasets/lang/p_e_50m.json"

  sd = SData(name, data)
  tdl, _ = sd.get_dls()
  for c, v in tdl.take(10):
    if len(c) > 0:
      print(c, v)
