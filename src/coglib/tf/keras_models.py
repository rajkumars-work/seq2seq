import keras
import numpy as np

MAX_TOKENS = 128


class PositionalEmbedding(keras.layers.Layer):
  def __init__(self, vocab_size, d_model, embed_init="uniform"):
    super().__init__()
    self.d_model = d_model
    self.embedding = keras.layers.Embedding(
      vocab_size, d_model, mask_zero=True, embeddings_initializer=embed_init
    )
    self.pos_encoding = self.positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = keras.ops.shape(x)[1]
    x = self.embedding(x)
    # This sets the scale of the embedding and positonal_encoding
    x *= keras.ops.sqrt(keras.ops.cast(self.d_model, float))
    x = x + self.pos_encoding[np.newaxis, :length, :]
    return x

  def positional_encoding(self, length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return keras.ops.cast(pos_encoding, dtype=float)


class BaseAttention(keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = keras.layers.LayerNormalization()
    self.add = keras.layers.Add()


class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
      query=x, key=context, value=context, return_attention_scores=True
    )

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x


class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(query=x, value=x, key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x


class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x


class FeedForward(keras.layers.Layer):
  def __init__(self, d_model, dff=None, dropout_rate=0.1, kernel_init="glorot_uniform"):
    super().__init__()
    dff = dff if dff else d_model * 4
    self.seq = keras.Sequential(
      [
        keras.layers.Dense(dff, activation="relu", kernel_initializer=kernel_init),
        keras.layers.Dense(d_model, kernel_initializer=kernel_init),
        keras.layers.Dropout(dropout_rate),
      ]
    )
    self.add = keras.layers.Add()
    self.layer_norm = keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x


class EncoderLayer(keras.layers.Layer):
  def __init__(
    self,
    *,
    d_model,
    num_heads,
    dff=None,
    dropout_rate=0.1,
    kernel_init="glorot_uniform",
  ):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
      num_heads=num_heads,
      key_dim=d_model,
      dropout=dropout_rate,
      kernel_initializer=kernel_init,
    )

    self.ffn = FeedForward(
      d_model, dff, dropout_rate=dropout_rate, kernel_init=kernel_init
    )

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x


class Encoder(keras.layers.Layer):
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
    self.dropout = keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(keras.layers.Layer):
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
      key_dim=d_model,
      dropout=dropout_rate,
      kernel_initializer=kernel_init,
    )
    self.cross_attention = CrossAttention(
      num_heads=num_heads,
      key_dim=d_model,
      dropout=dropout_rate,
      kernel_initializer=kernel_init,
    )
    self.ffn = FeedForward(
      d_model, dff, dropout_rate=dropout_rate, kernel_init=kernel_init
    )

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)
    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x


class Decoder(keras.layers.Layer):
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
    self.dropout = keras.layers.Dropout(dropout_rate)
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
  def call(self, x, context):
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
    x = self.dropout(x)
    for i in range(self.num_layers):
      x = self.dec_layers[i](x, context)
    return x  # The shape of x is (batch_size, target_seq_len, d_model)).


class Transformer(keras.Model):
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

    self.final_layer = keras.layers.Dense(
      target_vocab_size, kernel_initializer=kernel_init
    )

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the # first argument.
    context, x = inputs
    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      del logits._keras_mask
    except AttributeError:
      pass

    return logits

  def get_config(self):
    d = {
      "num_layers": self.num_layers,
      "num_heads": self.num_heads,
      "d_model": self.d_model,
      "input_vocab_size": self.input_vocab_size,
      "target_vocab_size": self.target_vocab_size,
    }
    return d


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model=128, warmup_steps=4000):
    super().__init__()

    self.d_model = keras.ops.cast(d_model, float)
    self.warmup_steps = warmup_steps

  def get_config(self):
    d = {"d_model": self.d_model, "warmup_steps": self.warmup_steps}
    return d

  def __call__(self, step):
    step = keras.ops.cast(step, dtype=float)
    arg1 = keras.ops.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return keras.ops.rsqrt(self.d_model) * keras.ops.minimum(arg1, arg2)


class CGenerator(keras.Model):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, output_sentence, max_length=MAX_TOKENS, training=False):
    sentence = [sentence]
    output_sentence = [output_sentence]

    # tokenize context and prompt sentences. Leave out the prompt end token
    sentence = self.tokenizers.context.tokenize(sentence).to_tensor()
    encoder_input = sentence

    # Note; For now we are not using the batch dimension. Not sure if it will be more efficient
    output_sentence = self.tokenizers.target.tokenize(output_sentence)[0]
    if training is False:
      output_sentence = output_sentence[:-1]  # leave out the '[END]' token
    olen = len(output_sentence)

    # Get the end token so we know when we are done with generation
    start_end = self.tokenizers.target.tokenize([""])[0]
    end = start_end[1][np.newaxis]

    output_array = []
    # for i in keras.ops.range(olen):
    for i in keras.ops.arange(olen):
      index = output_sentence[i][np.newaxis]
      output_array.append(index)

    for i in keras.ops.arange(max_length):
      output = keras.ops.convert_to_tensor(output_array, dtype="int32")
      output = keras.ops.reshape(output, (1, len(output)))
      predictions = self.transformer([encoder_input, output], training=training)

      # Select the last token from the `seq_len` dimension
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`

      predicted_ids = keras.ops.argmax(predictions, axis=-1)
      predicted_id = predicted_ids[0]

      # Concatenate the `predicted_id` to the output which is given to decoder as input
      output_array.append(predicted_id)

      predicted_id = keras.ops.cast(predicted_ids, dtype=end.dtype)
      if predicted_id == end:
        break

    # The output shape is `(1, tokens)`
    output = keras.ops.convert_to_tensor(output_array, dtype="int32")
    output = keras.ops.reshape(output, (1, len(output)))
    text = self.tokenizers.target.detokenize(output)[0]  # Shape: `()`
    tokens = self.tokenizers.target.lookup(output)[0]

    return text, tokens

  def get_config(self):
    d = {
      "tokenizers": self.tokenizers,
      "transformer": self.transformer,
    }
    return d


class ExportCGenerator(keras.Model):
  def __init__(self, cgenerator):
    self.cgenerator = cgenerator

  def __call__(self, sentence, target_sentence, training=False):
    (result, tokens) = self.cgenerator(
      sentence, target_sentence, max_length=MAX_TOKENS, training=training
    )
    return result

  def get_config(self):
    d = {"cgenerator": self.cgenerator}
    return d


# - routines ---
def masked_loss(label, pred):
  mask = label != 0
  loss_object = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
  )
  loss = loss_object(label, pred)

  mask = keras.ops.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = keras.ops.sum(loss) / keras.ops.sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = keras.ops.argmax(pred, axis=2)
  label = keras.ops.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = keras.ops.cast(match, dtype=float)
  mask = keras.ops.cast(mask, dtype=float)
  return keras.ops.sum(match) / keras.ops.sum(mask)
