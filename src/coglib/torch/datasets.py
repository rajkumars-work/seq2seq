import json
import logging
from itertools import islice
from time import time
from typing import List

from torch.utils.data import Dataset

from ..utils.preprocess import get_prep, sp_prep
from ..utils.text_io import Max_Rows, download, file_for_index, prep_sizes, vocab_files

Max_length = 1024
Batch_size = 32
Eval_batches = 100


# - Cogs datatsets ------------------------------------------------------
# Cogs memory dataset
class CogsDS(Dataset):
  def __init__(self, name: str, data: str, start: int = 0, stop: int = None):
    super(CogsDS).__init__()
    cd = CogsData(name, data, start, stop)
    self.data = list(cd.data())

  def __getitem__(self, idx):
    return self.data[idx]

  def __len__(self):
    return len(self.data)


# Cogs distributed dataset
class CogsDDS(Dataset):
  Max_Files = 2
  Empty = ("", "")

  def __init__(self, name: str, start: int = 0, stop: int = None):
    super(CogsDDS).__init__()
    self.name = name
    self.start = start
    self.stop = stop
    self.files, self.count = self.file_counts()
    self.items = {}
    self.time = {}

  def __getitem__(self, idx):
    index = idx + self.start
    if index > self.count or (self.stop and index > self.stop):
      return CogsDDS.Empty
    i, o = divmod(index, Max_Rows)
    return self.lookup(i, o)

  def __len__(self):
    return self.count

  def lookup(self, index: int, offset: int):
    if index not in self.items:
      self.load(index)
    res = CogsDDS.Empty
    if (index in self.items) and (len(self.items[index]) > offset):
      res = self.items[index][offset]
    return res if len(res) == 2 else CogsDDS.Empty

  def load(self, index):
    fn = file_for_index(self.name, index)
    if fn in self.files:
      with open(fn, "r") as fh:
        ds = (json.loads(li) for li in fh)
        self.items[index] = [(d["context"], d["target"]) for d in ds]
        self.time[index] = int(time())
    self.purge()

  # keep only Max_Items files open
  def purge(self):
    if len(self.items) > CogsDDS.Max_Files:
      i, _ = sorted(self.time.items(), key=lambda i: i[1])[0]
      logging.info(f"Torch DS loader: Evicting index {i} of {len(self.items)}")
      self.items[i] = None
      self.items.pop(i, None)
      self.time.pop(i, None)

  def file_counts(self):
    count = 0
    files = {}
    for k, v in prep_sizes(self.name).items():
      count = count + v
      files[k] = count
    return files, count


# ---------------------------
class CogsData:
  def __init__(self, name: str, start: int = 0, stop: int = None):
    super(CogsDS).__init__()
    self.name = name
    self.start = start
    self.stop = stop

  def data(self):
    return self.get_it()

  def get_it(self):
    it = islice(get_prep(self.name), self.start, self.stop)
    it = ((d["context"], d["target"]) for d in it)
    return it

  def write_vocab_file(self, filepath: str, vocab: List[str]):
    with open(filepath, "w") as f:
      for token in vocab:
        print(token, file=f)

  def generate_vocabs(self, name: str, data: str):
    from ..utils.tokenizer import ReservedTokens

    download(name, data)
    c_vocab, t_vocab = sp_prep(name, data)

    extras = ReservedTokens
    context_vocab_file, target_vocab_file = vocab_files(name)
    self.write_vocab_file(context_vocab_file, extras + c_vocab)
    self.write_vocab_file(target_vocab_file, extras + t_vocab)


if __name__ == "__main__":
  name = "por_en_torch"
  ds = CogsDDS(name)
  print(next(iter(ds)))
  print(next(iter(ds)))
