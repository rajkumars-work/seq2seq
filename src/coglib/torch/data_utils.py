from typing import Tuple

import torch
from test_dataset import TestDataset
from torch import Tensor
from torch.utils.data import dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator

# - Data -----------------

# - Utils --
batch_size = 10
chunk_size = 5
max_tokens = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer(None)


def generate_square_subsequent_mask(sz: int) -> Tensor:
  return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


def batchify(data: Tensor, bsz: int) -> Tensor:
  seq_len = data.size(0) // bsz
  data = data[: seq_len * bsz]
  data = data.view(bsz, seq_len).t().contiguous()
  return data.to(device)


def data_process(raw_text_iter: dataset.IterableDataset, vocab: Vocab) -> Tensor:
  data = [
    torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter
  ]
  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
  seq_length = min(chunk_size, len(source) - 1 - i)
  data = source[i : i + seq_length]
  target = source[i + 1 : i + 1 + seq_length].reshape(-1)
  return data, target


# - Vocab--
def get_vocab() -> Vocab:
  it = TestDataset()
  vocab = build_vocab_from_iterator(
    map(tokenizer, it), specials=["<unk>"], max_tokens=max_tokens
  )
  vocab.set_default_index(vocab["<unk>"])
  return vocab


def get_batched_data(vocab: Vocab) -> Tensor:
  it = TestDataset()
  data = data_process(it, vocab)
  batched = batchify(data, batch_size)
  return batched


def get_data_target(data: Tensor) -> Tuple[Tensor, Tensor]:
  return (get_batch(data, i) for i in range(0, data.size(0) - 1, chunk_size))


if __name__ == "__main__":
  vocab = get_vocab()
  print(vocab.get_stoi())
  print(vocab.get_itos())

  data = get_batched_data(vocab)
  print(data)
  print(data.shape)

  dts = get_data_target(data)
  for i, (d, t) in enumerate(dts):
    print(f"---{i}----")
    print(d)
    print(t)
