import math
from functools import partial
from typing import Dict, Iterator

import tensorflow as tf

from ..utils.prep import Prep

# from ..utils.preprocess import get_processed
from ..utils.preprocess import get_prep


def device_count():
  from tensorflow.python.client import device_lib

  device_count = max(len(device_lib.list_local_devices()), 1)
  device_count = 2 ** int(math.log2(device_count))
  return device_count


class TData:
  BATCH_SIZE = 64
  VAL_ROWS = 1000
  UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
  BOS, EOS = "[START]", "[END]"
  Buffer_Size = 50000
  Device_Batch_Size = 32
  Batch_Size = Device_Batch_Size * device_count()
  Val_Batches = 10
  Val_Size = Val_Batches * Batch_Size
  Max_Tokens = 128

  def __init__(self, name: str, data: str, uri_prep=True):
    self.name = name
    self.data = data
    self.pr = Prep(name, data, torch=False, uri_prep=uri_prep)
    self.ct = self.pr.ct
    self.tt = self.pr.tt
    self.const_length = True

    self.ds_signature = {
      "context": tf.TensorSpec(shape=(), dtype=tf.string),
      "target": tf.TensorSpec(shape=(), dtype=tf.string),
    }

  def get_vocab_sizes(self):
    return self.ct.vocab_size(), self.tt.vocab_size()

  def get_dls(self):
    return self.get_batched_data()

  def get_dss(self):
    return ((d["context"], d["target"]) for d in self._get_it())

  def get_batched_data(self):
    examples = self._create_ds()
    batches = self._make_batches(examples)
    eval_bds = batches.take(TData.Val_Batches)
    train_bds = batches.skip(TData.Val_Batches)
    return train_bds, eval_bds

  def _create_ds(self):
    tp = partial(self._get_it)
    tds = tf.data.Dataset.from_generator(tp, output_signature=self.ds_signature)
    return tds.map(lambda d: (d["context"], d["target"]), tf.data.AUTOTUNE)

  def _get_it(self) -> Iterator[Dict[str, str]]:
    return get_prep(self.name)

  def _make_batches(self, ds):
    return (
      ds.shuffle(TData.Buffer_Size)
      .batch(TData.Batch_Size, drop_remainder=True)
      .map(lambda x, y: self._prepare_batch(x, y), tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

  # tokenize, trim and zero-pad
  # if const_lenght, shape is always [batch_size, MAX_TOKENS] else [batch_size, max_length]
  def _prepare_batch(self, ct, tt):
    len = TData.Max_Tokens if self.const_length else None
    oshape = [None, len]  # out shape [batch,len]

    ct = self.ct.tokenize(ct)  # Output is ragged.
    ct = ct[:, : TData.Max_Tokens]  # Trim to MAX_TOKENS.
    ct = ct.to_tensor(shape=oshape)  # Convert to 0-padded dense Tensor

    tt = self.tt.tokenize(tt)
    tt = tt[:, : (TData.Max_Tokens + 1)]
    tt_inputs = tt[:, :-1].to_tensor(shape=oshape)  # Drop the [END] tokens
    tt_labels = tt[:, 1:].to_tensor(shape=oshape)  # Drop the [START] tokens

    return (ct, tt_inputs), tt_labels

  def get_transforms(self):
    def st(s):
      return self.to_tensor(self.ct.tokenize(s, mark=False))

    return st, self.pr.tt.vocab, self.pr.encode_context

  def encode_context(self, cs: str) -> str:
    return " ".join(self.cm.encode_as_pieces(cs))


if __name__ == "__main__":
  name = "por_en_new"
  data = "ml-sketchbook.cogs_data.por_en"

  td = TData(name, data)
  pr = td.pr
  print(pr, pr.ct)
