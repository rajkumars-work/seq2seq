from typing import Iterable, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k, multi30k
from torchtext.vocab import build_vocab_from_iterator

from .utils import device

DEVICE = device
print("Device: ", device)

BATCH_SIZE = 128

# Define special symbols and indices
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

multi30k.URL["train"] = (
  "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
)
multi30k.URL["valid"] = (
  "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
)
SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"

# Place-holders
token_transform = {}
vocab_transform = {}


token_transform[SRC_LANGUAGE] = get_tokenizer("spacy", language="de_core_news_sm")
token_transform[TGT_LANGUAGE] = get_tokenizer("spacy", language="en_core_web_sm")


# -------------------------------------------------------
# function to collate data samples into batch tensors
def collate_fn(batch):
  src_batch, tgt_batch = [], []
  for src_sample, tgt_sample in batch:
    src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
    tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

  src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
  tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
  return src_batch, tgt_batch


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
  language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

  for data_sample in data_iter:
    yield token_transform[language](data_sample[language_index[language]])


# helper function to club together sequential operations
def sequential_transforms(*transforms):
  def func(txt_input):
    for transform in transforms:
      txt_input = transform(txt_input)
    return txt_input

  return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
  return torch.cat(
    (torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
  )


# - ------------------------------------------
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  train_iter = Multi30k(split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
  vocab_transform[ln] = build_vocab_from_iterator(
    yield_tokens(train_iter, ln),
    min_freq=1,
    specials=special_symbols,
    special_first=True,
  )

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)


# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  text_transform[ln] = sequential_transforms(
    token_transform[ln],  # Tokenization
    vocab_transform[ln],  # Numericalization
    tensor_transform,
  )  # Add BOS/EOS and create tensor


# datasets and dataloaders
train_ds = Multi30k(split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

val_ds = Multi30k(split="valid", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)


def get_data():
  s_vs = len(vocab_transform[SRC_LANGUAGE])
  t_vs = len(vocab_transform[TGT_LANGUAGE])
  return train_dataloader, val_dataloader, s_vs, t_vs


def get_transforms():
  return text_transform[SRC_LANGUAGE], vocab_transform[TGT_LANGUAGE]


def post_process(r: str) -> str:
  return r
