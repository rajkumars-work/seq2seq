import re

import tensorflow as tf
from tensorflow.keras.layers import StringLookup

from ..utils.tokenizer import ReservedTokens


class SimpleTokenizer(tf.Module):
  START = tf.argmax(tf.constant(ReservedTokens) == "[START]")
  END = tf.argmax(tf.constant(ReservedTokens) == "[END]")
  bad_tokens = [re.escape(t) for t in ReservedTokens if t != "[UNK]"]
  bad_token_re = "|".join(bad_tokens)

  def __init__(self, vocab, const_length: int = None):
    self.const_length = const_length
    vocab = self.read_vocab(vocab) if isinstance(vocab, str) else vocab
    self.length = len(vocab)
    self.tokenizer = StringLookup(vocabulary=vocab)
    self.detokenizer = StringLookup(vocabulary=vocab, invert=True)

  def vocab_size(self) -> int:
    return self.length

  @tf.function
  def tokenize(self, strings):
    enc = tf.strings.split(strings)
    enc = self.tokenizer(enc)
    return self.add(enc)

  def add(self, enc):
    shape = enc.bounding_shape()[:-1]
    fill_shape = tf.concat([shape, tf.constant([1], dtype=tf.int64)], axis=0)
    starts = tf.fill(fill_shape, SimpleTokenizer.START)
    ends = tf.fill(fill_shape, SimpleTokenizer.END)
    res = tf.concat([starts, enc, ends], axis=1)
    return res

  def filter_text(self, token_txt):
    bad_cells = tf.strings.regex_full_match(token_txt, SimpleTokenizer.bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)
    return result

  @tf.function
  def detokenize(self, res):
    words = self.detokenizer(res)
    words = self.filter_text(words)
    result = tf.strings.reduce_join(words, separator=" ", axis=-1)
    return result

  @tf.function
  def lookup(self, tokens):
    return self.detokenizer(tokens)

  @tf.function
  def get_vocab_size(self):
    return self.tokenizer.vocabulary_size()

  def read_vocab(self, f):
    with open(f, "r") as fh:
      return [w.rstrip("\n") for w in fh]


if __name__ == "__main__":
  f = "/Users/rkumar/cogs/por_en/target.vocab"
  t = SimpleTokenizer(f, 10)

  ss = ["▁patient ▁clean ▁box", "patient clean box"]
  res = t.tokenize(ss)
  for r in res:
    print(r)
  for r in t.detokenize(res):
    print(r)
