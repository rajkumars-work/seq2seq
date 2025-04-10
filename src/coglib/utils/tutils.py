import itertools
import json
from typing import List

import numpy as np
import psutil
from more_itertools import chunked
from sentencepiece import SentencePieceProcessor

from ..utils.gcs import gfs
from .preprocess import DataPrep
from .sputils import sp_tokenizers
from .text_io import get_json

MAX_TOKENS = 128
Batch_Size = 32


def memory():
  return {"mem_pct": psutil.virtual_memory().percent}


# numpy array to python list-of-lists
def na_to_ls(a) -> str:
  a = a.numpy() if getattr(a, "numpy", None) else a
  return list(list(i.item() for i in r) for r in a)


def left_shift_and_pad(x):
  return np.pad(x[:, 1:], ((0, 0), (0, 1)), constant_values=0)


# label = x left shifted. x = x without last token
def add_label(x):
  label = x[:, 1:]
  x = x[:, :-1]
  return x, label


def reversed_batch(rows: int, cols: int, maxval: int, seed: int = None):
  if seed is not None:
    np.random.seed(seed)
  a = np.random.randint(low=0, high=maxval, size=[rows, cols])
  r = a[:, -1::-1]
  ra = np.hstack([r, a])
  return a, ra


def reversed_batches(batches: int, rows: int, cols: int, maxval: int, seed: int = 0):
  np.random.seed(seed)
  return (reversed_batch(rows, cols, maxval) for _ in range(batches))


# - generate data ---
def tokenize_na(x: List[str], t: SentencePieceProcessor, const_length: int = None):
  x = [[t.bos_id()] + t.encode(s) + [t.eos_id()] for s in x]
  max_len = const_length if const_length else max(len(s) for s in x)
  max_len = min(max_len, MAX_TOKENS)
  x = [a + ([0] * max(max_len - len(a), 0)) for a in x]
  x = [a[:max_len] for a in x]
  return np.array(x, dtype=np.int32)


def get_batched_data(
  name: str, data: str, bs: int = Batch_Size, const_length: int = None
):
  csp, tsp = DataPrep(name, data).tokenizers()
  for c in chunked(get_json(name, data), bs):
    if len(c) == bs:
      cs = [r["context"] for r in c]
      ts = [r["target"] for r in c]
      ca = tokenize_na(cs, csp, const_length)
      ta = tokenize_na(ts, tsp, const_length)
      ta, labels = add_label(ta)
      yield (ca, ta), labels


def _process_chunk(c: List[str], cols: int):
  ca = [[int(s) for s in e["context"]] for e in c]
  ca = [c + [0] * max(cols - len(c), 0) for c in ca]
  ca = [c[:cols] for c in ca]
  ca = np.array(ca, dtype=np.int32)
  tcols = cols + 1  # make target longer as we truncate when creating label
  ta = [[int(s) for s in e["target"]] for e in c]
  ta = [c + [0] * max(tcols - len(c), 0) for c in ta]
  ta = [c[:tcols] for c in ta]
  ta = np.array(ta, dtype=np.int32)
  ta, labels = add_label(ta)
  return (ca, ta), labels


def test_batches(rows: int = Batch_Size, cols: int = MAX_TOKENS, big=False):
  name = (
    "gs://cogs-data/testing/por_en/tokens.jsons"
    if big
    else "resources/por_en/tokens.jsons"
  )
  return get_test_batches(name, rows, cols)


def get_test_batches(name: str, rows: int, cols: int):
  fname = name
  if name.startswith("gs://"):
    fname = "/tmp/tokens.jsons"
    gfs().copy(name, fname)

  with open(fname, "r") as fh:
    while fh:
      lines = list(itertools.islice((json.loads(li) for li in fh), rows))
      if len(lines) == rows:
        yield _process_chunk(lines, cols)


# -
# test datasets
#
def gen_test_batches(name: str, data: str):
  csp, tsp = sp_tokenizers(name)
  tokens = (
    {"context": csp.encode(e["context"]), "target": tsp.encode(e["target"])}
    for e in get_json(name, data)
  )
  chunks = chunked(tokens, Batch_Size)
  batches = (_process_chunk(b, MAX_TOKENS) for b in chunks)
  return batches


def gen_tf_batches(name: str, data: str, bs: int):
  from ..tf.tf_v1_ds import get_test_data
  from ..tf.vocab import tokenizers

  tzs = tokenizers(name, data)
  return get_test_data(name, data, tzs, bs)


def write_test_batches(name: str, data: str, bs: int, fn: str, n: int = 0):
  ds = get_batched_data(name, data, bs)
  ds = itertools.islice(ds, n) if n else ds
  write_ds(ds, fn)


def write_tf_batches(name: str, data: str, bs: int, fn: str, n: int = 0):
  ds = gen_tf_batches(name, data, bs)
  ds = ds.take(n) if n else ds
  write_ds(ds, fn)


# io of datastreams (iterator-of-numpy or datasets)
def write_ds(ds, fn: str):
  with gfs().open(fn, "w") as fh:
    for (c, t), la in ds:
      d = {
        "context": na_to_ls(c),
        "target": na_to_ls(t),
        "labels": na_to_ls(la),
      }
      s = json.dumps(d)
      fh.write(s + "\n")


def read_ds(fn: str, dtype=np.int32):
  fname = fn
  if fn.startswith("gs://"):
    fname = "/tmp/tokens.jsons"
    if not gfs().exists(fname):
      gfs().copy(fn, fname)

  with open(fname, "r") as fh:
    for li in fh:
      d = json.loads(li)
      c = np.array(d["context"], dtype=dtype)
      t = np.array(d["target"], dtype=dtype)
      ls = np.array(d["labels"], dtype=dtype)
      yield (c, t), ls
  return
