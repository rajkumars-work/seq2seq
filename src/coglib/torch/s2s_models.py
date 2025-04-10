import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer

from .s2s_data import SData

BOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX = (
  SData.BOS_IDX,
  SData.EOS_IDX,
  SData.PAD_IDX,
  SData.UNK_IDX,
)
BOS, EOS = SData.BOS, SData.EOS


class PositionalEncoding(nn.Module):
  def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
    super(PositionalEncoding, self).__init__()
    den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
    pos = torch.arange(0, maxlen).reshape(maxlen, 1)
    pos_embedding = torch.zeros((maxlen, emb_size))
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    pos_embedding = pos_embedding.unsqueeze(-2)

    self.dropout = nn.Dropout(dropout)
    self.register_buffer("pos_embedding", pos_embedding)

  def forward(self, token_embedding: Tensor):
    return self.dropout(
      token_embedding + self.pos_embedding[: token_embedding.size(0), :]
    )


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size: int, emb_size):
    super(TokenEmbedding, self).__init__()
    self.embedding = nn.Embedding(vocab_size * 2, emb_size)
    self.emb_size = emb_size

  def forward(self, tokens: Tensor):
    return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
  def __init__(
    self,
    num_layers: int,
    d_model: int,
    num_heads: int,
    dff: int,
    dropout_rate: float,
    input_vocab_size: int,
    target_vocab_size: int,
  ):
    super(Seq2SeqTransformer, self).__init__()
    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size
    self.transformer = Transformer(
      d_model=d_model,
      nhead=num_heads,
      num_encoder_layers=num_layers,
      num_decoder_layers=num_layers,
      dim_feedforward=dff,
      dropout=dropout_rate,
    )
    self.generator = nn.Linear(d_model, target_vocab_size)
    self.src_tok_emb = TokenEmbedding(input_vocab_size, d_model)
    self.tgt_tok_emb = TokenEmbedding(target_vocab_size, d_model)
    self.positional_encoding = PositionalEncoding(d_model, dropout=dropout_rate)

  def forward(
    self,
    src: Tensor,
    trg: Tensor,
    src_mask: Tensor,
    tgt_mask: Tensor,
    src_padding_mask: Tensor,
    tgt_padding_mask: Tensor,
    memory_key_padding_mask: Tensor,
  ):
    src_emb = self.positional_encoding(self.src_tok_emb(src))
    tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
    outs = self.transformer(
      src_emb,
      tgt_emb,
      src_mask,
      tgt_mask,
      None,
      src_padding_mask,
      tgt_padding_mask,
      memory_key_padding_mask,
    )
    return self.generator(outs)

  def encode(self, src: Tensor, src_mask: Tensor):
    return self.transformer.encoder(
      self.positional_encoding(self.src_tok_emb(src)), src_mask
    )

  def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
    return self.transformer.decoder(
      self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
    )


# - ------------------------------------------------------
def generate_square_subsequent_mask(sz, device):
  mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
  mask = (
    mask.float()
    .masked_fill(mask == 0, float("-inf"))
    .masked_fill(mask == 1, float(0.0))
  )
  return mask


def create_mask(src, tgt, device):
  src_seq_len = src.shape[0]
  tgt_seq_len = tgt.shape[0]

  tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
  src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

  src_padding_mask = (src == PAD_IDX).transpose(0, 1)
  tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
  return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def accuracy(logits, target) -> float:
  la = torch.argmax(logits, -1)
  e = torch.sum((la == target) & (target != PAD_IDX))
  z = torch.sum(target != PAD_IDX)
  res = e / z
  return torch.tensor(0.0) if math.isinf(res) else res


def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
  src = src.to(device)
  src_mask = src_mask.to(device)

  memory = model.encode(src, src_mask)
  ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
  for i in range(max_len - 1):
    memory = memory.to(device)
    tgt_mask = (
      generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)
    ).to(device)
    out = model.decode(ys, memory, tgt_mask)
    out = out.transpose(0, 1)
    prob = model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.item()

    ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
    if next_word == EOS_IDX:
      break
  return ys


# function to translate input sentence into target language
def translate(
  model: torch.nn.Module,
  src_sentence: str,
  src_text_transform,
  tgt_vocab_transform,
  device,
):
  model.eval()
  src = src_text_transform(src_sentence).view(-1, 1)
  num_tokens = src.shape[0]
  src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
  tgt_tokens = greedy_decode(
    model,
    src,
    src_mask,
    max_len=num_tokens + 5,
    start_symbol=BOS_IDX,
    device=device,
  ).flatten()
  tokens = list(tgt_tokens.cpu().numpy())
  tokens = tgt_vocab_transform.lookup_tokens(tokens)
  os = " ".join(tokens).replace(BOS, "").replace(EOS, "")
  return os


def greedy_predict(model, src, src_mask, max_len, tgt, device):
  from .s2s_models import generate_square_subsequent_mask

  src = src.to(device)
  src_mask = src_mask.to(device)

  memory = model.encode(src, src_mask)

  ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(device)
  for j in range(len(tgt)):
    next_word = tgt[j, 0]
    ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

  for i in range(max_len - 1):
    memory = memory.to(device)
    tgt_mask = (
      generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)
    ).to(device)
    out = model.decode(ys, memory, tgt_mask)
    out = out.transpose(0, 1)
    prob = model.generator(out[:, -1])
    _, next_word = torch.max(prob, dim=1)
    next_word = next_word.item()

    ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
    if next_word == EOS_IDX:
      break
  return ys


# function to translate input sentence into target language
def predict(
  model: torch.nn.Module,
  src_sentence: str,
  tgt_sentence: str,
  src_text_transform,
  tgt_text_transform,
  tgt_vocab_transform,
  device,
):
  model.eval()
  src = src_text_transform(src_sentence).view(-1, 1)
  tgt = tgt_text_transform(tgt_sentence).view(-1, 1)
  num_tokens = src.shape[0]
  tgt_tokens = tgt.shape[0]
  src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
  tgt_tokens = greedy_predict(
    model,
    src,
    src_mask,
    max_len=num_tokens + 5,
    tgt=tgt,
    device=device,
  ).flatten()
  tokens = list(tgt_tokens.cpu().numpy())
  # tokens = [t for t in tokens if t not in [BOS_IDX, EOS_IDX]]
  tokens = [t for t in tokens if t not in [BOS_IDX, EOS_IDX, UNK_IDX]]
  tokens = tgt_vocab_transform.lookup_tokens(tokens)
  return " ".join(tokens)
