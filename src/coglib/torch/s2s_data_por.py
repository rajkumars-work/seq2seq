import pathlib
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ..utils.preprocess import DataPrep, sp_preprocess
from ..utils.sputils import create_models
from ..utils.text_io import download, vocab_files
from .datasets import CogsDataset
from .simple_tokenizer import SimpleTokenizer

BATCH_SIZE = 128

# Define special symbols and indices
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

name = "por_en"
data = "ml-sketchbook.cogs_data.por_en"


SRC_LANGUAGE = "por"
TGT_LANGUAGE = "en"


def generate_vocabs(self, name: str, data: str):
  download(name, data)
  c_vocab, t_vocab = sp_preprocess(name, data)

  extras = SimpleTokenizer.Reserved_tokens
  context_vocab_file, target_vocab_file = vocab_files(name)
  self.write_vocab_file(context_vocab_file, extras + c_vocab)
  self.write_vocab_file(target_vocab_file, extras + t_vocab)


def get_tokenizers(name, data):
  download(name, data)
  create_models(name, data)
  context_vocab_file, target_vocab_file = vocab_files(name)
  cvf = pathlib.Path(context_vocab_file).is_file()
  tvf = pathlib.Path(target_vocab_file).is_file()
  if not (cvf and tvf):
    generate_vocabs(name, data)
  ct = SimpleTokenizer(context_vocab_file)
  tt = SimpleTokenizer(target_vocab_file)

  return ct, tt


token_transform = {}
token_transform[SRC_LANGUAGE], token_transform[TGT_LANGUAGE] = get_tokenizers(
  name, data
)


# -------------------------------------------------------
# function to collate data samples into batch tensors
# returns ([s_src_len, batch_size], [t_src_len, batch_size])
def tensor_transform(token_ids: List[int]):
  return torch.cat(
    (
      torch.tensor([BOS_IDX], dtype=torch.int),
      torch.tensor(token_ids, dtype=torch.int),
      torch.tensor([EOS_IDX], dtype=torch.int),
    )
  )


def collate_fn(batch):
  # function to add BOS/EOS and create tensor for input sequence indices

  src_batch = [
    tensor_transform(token_transform[SRC_LANGUAGE].tokenize(s, mark=False))
    for s, _ in batch
  ]

  tgt_batch = [
    tensor_transform(token_transform[TGT_LANGUAGE].tokenize(t, mark=False))
    for _, t in batch
  ]
  src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
  tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
  return src_batch, tgt_batch


def get_data():
  # datasets and dataloaders
  train_ds = CogsDataset(name, data, 1000, loop=True)
  train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

  val_ds = CogsDataset(name, data, 0, 1000, loop=True)
  val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

  src_vs = token_transform[SRC_LANGUAGE].vocab_size()
  tgt_vs = token_transform[TGT_LANGUAGE].vocab_size()
  return train_dataloader, val_dataloader, src_vs, tgt_vs


def get_transforms():
  def st(s):
    ids = tensor_transform(token_transform[SRC_LANGUAGE].tokenize(s, mark=False))
    return tensor_transform(ids)

  tt = token_transform[TGT_LANGUAGE].vocab
  return st, tt


dp = DataPrep(name, data)


def post_process(r: str) -> str:
  return dp.decode_targets(r)


if __name__ == "__main__":
  r, e, _, _ = get_data()
  for k in range(3):
    j = 0
    for i in e:
      j += 1
      print(j)
    print(j)
