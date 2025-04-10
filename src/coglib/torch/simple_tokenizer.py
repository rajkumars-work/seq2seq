from collections import OrderedDict
from typing import List

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

from ..utils.tokenizer import UNK_IDX, End, Pad, Remove_tokens, ReservedTokens, Start


class SimpleTokenizer:
  def __init__(self, voc, const_length: int = None):
    self.const_length = const_length
    vs = self.read_vocab(voc) if isinstance(voc, str) else voc
    self.vocab = vocab(OrderedDict([(t, 1) for t in vs]), specials=ReservedTokens)
    self.vocab.set_default_index(UNK_IDX)
    self.tokenizer = get_tokenizer(None)

  def vocab_size(self) -> int:
    return self.vocab.__len__()

  def tokenize(self, s, mark: bool = True):
    token = self.tokenizer(s)
    token = [Start] + token + [End] if mark else token
    if self.const_length:
      token = token[: self.const_length]
      token = token + [Pad] * max((self.const_length - len(token)), 0)
    token = self.vocab(token)
    return token

  # tokenize list of strings (tokens padded to make length equal)
  def tokenizes(self, ss, pad: bool = True):
    tokens = [[Start] + self.tokenizer(s) + [End] for s in ss]
    return self.pad_batch(tokens) if pad else tokens

  def pad_batch(self, batch):
    batch_length = max(len(tokens) for tokens in batch)
    batch_length = self.const_length if self.const_length else batch_length
    batch = [tokens[:batch_length] for tokens in batch]
    batch = [tokens + [Pad] * max((batch_length - len(tokens)), 0) for tokens in batch]
    batch = [self.vocab(tokens) for tokens in batch]
    return batch

  def detokenize(self, tokens=List[int]) -> str:
    words = self.vocab.lookup_tokens(tokens)
    words = [w for w in words if w not in Remove_tokens]
    return " ".join(words)

  def detokenizes(self, tokenls=List[List[int]]) -> str:
    return [self.detokenize(tokens) for tokens in tokenls]

  def read_vocab(self, f):
    with open(f, "r") as fh:
      return [w.rstrip("\n") for w in fh]
