import numpy as np
import torch
from torch import nn

from ..params import Max_Tokens
from .utils import device


class PositionalEmbedding(nn.Module):
  def __init__(self, vocab_size, d_model, embed_init="uniform"):
    super().__init__()
    self.d_model = d_model
    self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
    self.embedding.weight.data.uniform_()  # initializer
    pos_encoding = self.positional_encoding(length=2048, depth=d_model)
    self.register_buffer("pos_encoding", pos_encoding)

  def forward(self, x):
    length = x.shape[1]
    x = self.embedding(x)
    # This sets the scale of the embedding and positonal_encoding
    x *= np.sqrt(self.d_model)
    x = x + self.pos_encoding[np.newaxis, :length, :]
    x = x.to(torch.float)
    return x

  def positional_encoding(self, length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return torch.from_numpy(pos_encoding).to(torch.float)


class BaseAttention(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.layernorm = nn.LayerNorm(kwargs["embed_dim"]).to(device)  # kdim = d_model
    self.mha = nn.MultiheadAttention(**kwargs, batch_first=True).to(device)


class CrossAttention(BaseAttention):
  def forward(self, x, context):
    attn_output, _ = self.mha(query=x, key=context, value=context)
    x = x + attn_output
    x = self.layernorm(x)
    return x


class GlobalSelfAttention(BaseAttention):
  def forward(self, x):
    attn_output, _ = self.mha(query=x, value=x, key=x)
    x = x + attn_output
    x = self.layernorm(x)
    return x


class CausalSelfAttention(BaseAttention):
  def forward(self, x):
    attn_output, _ = self.mha(query=x, value=x, key=x)
    x = x + attn_output
    x = self.layernorm(x)
    return x


class FeedForward(nn.Module):
  def __init__(self, d_model, dff=None, dropout_rate=0.1, kernel_init="glorot_uniform"):
    super().__init__()
    dff = dff if dff else d_model * 4

    l1 = nn.Linear(d_model, dff)
    torch.nn.init.xavier_uniform(l1.weight)

    l2 = nn.Linear(dff, d_model)
    torch.nn.init.xavier_uniform(l2.weight)

    self.seq = nn.Sequential(
      l1,
      nn.ReLU(),
      l2,
      nn.Dropout(dropout_rate),
    ).to(device)
    self.layer_norm = nn.LayerNorm(d_model).to(device)

  def forward(self, x):
    x = x + self.seq(x)
    x = self.layer_norm(x)
    return x


class EncoderLayer(nn.Module):
  def __init__(
    self,
    *,
    d_model,
    num_heads,
    dff=None,
    dropout_rate=0.1,
    kernel_init="xavier_uniform",
  ):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
      num_heads=num_heads,
      embed_dim=d_model,
      dropout=dropout_rate,
    )

    self.ffn = FeedForward(
      d_model, dff, dropout_rate=dropout_rate, kernel_init=kernel_init
    )

  def forward(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x


class Encoder(nn.Module):
  def __init__(
    self,
    *,
    num_layers,
    d_model,
    num_heads,
    vocab_size,
    dff=None,
    dropout_rate=0.1,
    kernel_init="glorot_uniform",
    embed_init="uniform",
  ):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
      vocab_size=vocab_size, d_model=d_model, embed_init=embed_init
    )

    self.enc_layers = [
      EncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout_rate,
        kernel_init=kernel_init,
      )
      for _ in range(num_layers)
    ]
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(nn.Module):
  def __init__(
    self,
    *,
    d_model,
    num_heads,
    dff=None,
    dropout_rate=0.1,
    kernel_init="glorot_uniform",
  ):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
      num_heads=num_heads,
      embed_dim=d_model,
      dropout=dropout_rate,
    )
    self.cross_attention = CrossAttention(
      num_heads=num_heads,
      embed_dim=d_model,
      dropout=dropout_rate,
    )
    self.ffn = FeedForward(
      d_model, dff, dropout_rate=dropout_rate, kernel_init=kernel_init
    )

  def forward(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)
    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x


class Decoder(nn.Module):
  def __init__(
    self,
    *,
    num_layers,
    d_model,
    num_heads,
    vocab_size,
    dff=None,
    dropout_rate=0.1,
    kernel_init="glorot_uniform",
    embed_init="uniform",
  ):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
      vocab_size=vocab_size, d_model=d_model, embed_init=embed_init
    )
    self.dropout = nn.Dropout(dropout_rate)
    self.dec_layers = [
      DecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout_rate,
        kernel_init=kernel_init,
      )
      for _ in range(num_layers)
    ]

  #  Note. context has already been encoded, x has not
  def forward(self, x, context):
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
    x = self.dropout(x)
    for i in range(self.num_layers):
      x = self.dec_layers[i](x, context)
    return x  # The shape of x is (batch_size, target_seq_len, d_model)).


class Transformer(nn.Module):
  def __init__(
    self,
    *,
    num_layers,
    num_heads,
    d_model,
    input_vocab_size,
    target_vocab_size,
    dff=None,
    dropout_rate=0.1,
    kernel_init="glorot_uniform",
    embed_init="uniform",
  ):
    super().__init__()

    self.num_layers = num_layers
    self.num_heads = num_heads
    self.d_model = d_model
    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size

    self.encoder = Encoder(
      num_layers=num_layers,
      num_heads=num_heads,
      d_model=d_model,
      dff=dff if dff else d_model * 4,
      vocab_size=input_vocab_size,
      dropout_rate=dropout_rate,
      kernel_init=kernel_init,
      embed_init=embed_init,
    )

    self.decoder = Decoder(
      num_layers=num_layers,
      d_model=d_model,
      num_heads=num_heads,
      dff=dff,
      vocab_size=target_vocab_size,
      dropout_rate=dropout_rate,
      kernel_init=kernel_init,
      embed_init=embed_init,
    )

    self.final_layer = nn.Linear(d_model, target_vocab_size)
    torch.nn.init.xavier_uniform(self.final_layer.weight)

  def forward(self, inputs):
    context, x = inputs
    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    return logits


# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
# sched = ScheduledOptim(optimizer, d_model=..., n_warmup_steps=...)
class ScheduledOptim:
  """A simple wrapper class for learning rate scheduling"""

  # def __init__(self, optimizer, d_model, n_warmup_steps, lr_mul=1.0):
  def __init__(self, optimizer, d_model, n_warmup_steps=4000, lr_mul=2.0):
    self._optimizer = optimizer
    self.lr_mul = lr_mul
    self.d_model = d_model
    self.n_warmup_steps = n_warmup_steps
    self.n_steps = 0

  def step_and_update_lr(self):
    "Step with the inner optimizer"
    self._update_learning_rate()
    self._optimizer.step()

  def zero_grad(self):
    "Zero out the gradients with the inner optimizer"
    self._optimizer.zero_grad()

  def _get_lr_scale(self):
    d_model = self.d_model
    n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
    arg1 = n_steps ** (-0.5)
    arg2 = n_steps * (n_warmup_steps ** (-1.5))
    return (d_model**-0.5) * min(arg1, arg2)

  def _update_learning_rate(self):
    """Learning rate scheduling per step"""

    self.n_steps += 1
    lr = self.lr_mul * self._get_lr_scale()

    for param_group in self._optimizer.param_groups:
      param_group["lr"] = lr


class CGenerator:
  def __init__(self, transformer, context_tokenizer, target_tokenizer):
    self.transformer = transformer.to(device)
    self.context_tokenizer = context_tokenizer
    self.target_tokenizer = target_tokenizer

  def predict(self, sentence, output_sentence, max_length=Max_Tokens):
    # tokenize context and prompt sentences. Leave out the prompt end token
    sentence = self.context_tokenizer.tokenize(sentence)
    context = torch.tensor(sentence, dtype=torch.int)
    context = torch.reshape(context, (1, len(context))).to(device)

    output_sentence = self.target_tokenizer.tokenize(output_sentence)
    output_sentence = output_sentence[:-1]  # leave out the '[END]' token
    output_array = output_sentence

    for i in torch.arange(max_length):
      output = torch.tensor(output_array, dtype=torch.int)
      output = torch.reshape(output, (1, len(output))).to(device)

      predictions = self.transformer((context, output))
      # Select the last token from the `seq_len` dimension
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`
      predicted_ids = torch.argmax(predictions, dim=-1)
      predicted_id = predicted_ids[0]

      # Concatenate the `predicted_id` to the output which is given to decoder as input
      output_array.append(predicted_id)

      if predicted_id.item() == self.target_tokenizer.EOS_IDX:
        break

    text = self.target_tokenizer.detokenize(output_array)
    return text


#
# - routines ---
# Note: more accurate tow work in log-domain
def masked_loss(label, pred):
  loss_object = nn.NLLLoss(reduction="none")
  m = nn.LogSoftmax(dim=-1)
  pred = m(pred)
  # pred logits shape = (batch, length, vocab_size). loss expects (batch, vocab_size, length), so
  loss = loss_object(torch.permute(pred, (0, 2, 1)), label.to(torch.long))

  mask = torch.where(label != 0, 1.0, 0.0)
  loss *= mask

  loss = torch.sum(loss) / torch.sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = torch.argmax(pred, dim=2)
  label = label.to(pred.dtype)
  match = label == pred

  mask = label != 0
  match = match & mask

  match = match.to(torch.float)
  mask = mask.to(torch.float)
  return torch.sum(match) / torch.sum(mask)
