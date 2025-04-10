from typing import List

import keras.ops as knp
import numpy as np
from keras.layers import StringLookup
from keras_nlp.tokenizers import Tokenizer

from ..utils.gcs import gfs
from ..utils.tokenizer import End, Remove_tokens, Start
from ..utils.tutils import split_na
from .simple_tokenizer import SimpleTokenizer


class KerasTokenizer(Tokenizer):
  def __init__(self, local_file_path: str, const_length: int = None):
    super().__init__()

    self.local_file_path = local_file_path
    self.const_length = const_length

    vocab = self.read_vocab(local_file_path)
    self.tokenizer = StringLookup(vocabulary=vocab)
    self.detokenizer = StringLookup(vocabulary=vocab, invert=True)

  def get_config(self):
    d = {
      "local_file_path": self.local_file_path,
      "const_length": self.const_length,
    }
    return d

  def tokenize(self, strings):
    iss = tensor_to_strings(strings)
    na = split_na(iss, Start, End, self.const_length)
    enc = self.tokenizer(na)
    return enc

  def detokenize(self, tokens):
    words = self.detokenizer(tokens)
    lists = knp.convert_to_numpy(words).tolist()
    lists = [[x.decode("utf-8") for x in li] for li in lists]
    lists = [[x for x in li if x not in Remove_tokens] for li in lists]
    ss = [" ".join(li).strip() for li in lists]
    return knp.convert_to_tensor(ss)

  def vocab_size(self):
    return self.tokenizer.vocabulary_size()

  def read_vocab(self, f) -> List[str]:
    with gfs().open(f) as fh:
      return [w.rstrip("\n") for w in fh]


# Keras utils
def tensor_to_strings(t) -> List[str]:
  arr = t.numpy() if getattr(t, "numpy", None) else t
  lis = [arr] if np.isscalar(arr) else arr.tolist()
  return [i.decode("utf-8") for i in lis]


if __name__ == "__main__":
  from ..utils import text_io

  name = "por_en_test"
  cpath, tpath = text_io.vocab_files(name)

  tsp = SimpleTokenizer(tpath)
  csp = SimpleTokenizer(cpath)

  e = "some text"
  e = knp.convert_to_tensor(e)
  tokens = tsp.tokenize(e)
  print(tokens)
  rre = tsp.detokenize(tokens)
  print(rre)
