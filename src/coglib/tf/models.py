import numpy as np
import tensorflow as tf

from ..params import Max_Tokens


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model, embed_init="uniform"):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(
      vocab_size, d_model, mask_zero=True, embeddings_initializer=embed_init
    )
    self.pos_encoding = self.positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This sets the scale of the embedding and positonal_encoding
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

  def positional_encoding(self, length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


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


class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff=None, dropout_rate=0.1, kernel_init="glorot_uniform"):
    super().__init__()
    dff = dff if dff else d_model * 4
    self.seq = tf.keras.Sequential(
      [
        tf.keras.layers.Dense(dff, activation="relu", kernel_initializer=kernel_init),
        tf.keras.layers.Dense(d_model, kernel_initializer=kernel_init),
        tf.keras.layers.Dropout(dropout_rate),
      ]
    )
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x


class EncoderLayer(tf.keras.layers.Layer):
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


class Encoder(tf.keras.layers.Layer):
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
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(tf.keras.layers.Layer):
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


class Decoder(tf.keras.layers.Layer):
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
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
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


class Transformer(tf.keras.Model):
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

    self.final_layer = tf.keras.layers.Dense(
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


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model=128, warmup_steps=4000):
    super().__init__()

    self.d_model = tf.cast(d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def get_config(self):
    d = {"d_model": self.d_model, "warmup_steps": self.warmup_steps}
    return d

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# lr = init_value / sqrt(step)
class SquarerootSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, init_value=0, warmup_steps=4000):
    super().__init__()

    self.init_value = tf.cast(init_value, dtype=tf.float32)
    self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    total = step + self.warmup_steps
    return self.init_value * tf.math.rsqrt(self.warmup_steps) / tf.math.rsqrt(total)


class CGenerator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, output_sentence, max_length=Max_Tokens, training=False):
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]
    assert isinstance(output_sentence, tf.Tensor)
    if len(output_sentence.shape) == 0:
      output_sentence = output_sentence[tf.newaxis]

    # tokenize context and prompt sentences. Leave out the prompt end token
    sentence = self.tokenizers.context.tokenize(sentence).to_tensor()
    encoder_input = sentence
    # Note; For now we are not using the batch dimension. Fix.
    output_sentence = self.tokenizers.target.tokenize(output_sentence)[0]
    if training is False:
      output_sentence = output_sentence[:-1]  # leave out the '[END]' token
    olen = len(output_sentence)

    # Get the end token so we know when we are done with generation
    start_end = self.tokenizers.target.tokenize([""])[0]
    end = start_end[1][tf.newaxis]

    # `tf.TensorArray` is required here, instead of a Python list, so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    for i in tf.range(olen):
      index = output_sentence[i][tf.newaxis]
      output_array = output_array.write(i, index)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions = self.transformer([encoder_input, output], training=training)

      # Select the last token from the `seq_len` dimension
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`

      predicted_id = tf.argmax(predictions, axis=-1)

      # Concatenate the `predicted_id` to the output which is given to decoder as input
      output_array = output_array.write(i + olen, predicted_id[0])

      if predicted_id == end:
        break

    # The output shape is `(1, tokens)`
    output = tf.transpose(output_array.stack())
    text = self.tokenizers.target.detokenize(output)[0]  # Shape: `()`
    tokens = self.tokenizers.target.lookup(output)[0]

    return text, tokens


class ExportCGenerator(tf.Module):
  def __init__(self, cgenerator, loss=0, acc=0, epochs=0):
    self.cgenerator = cgenerator
    self.loss = tf.Variable(loss)
    self.acc = tf.Variable(acc)
    self.epochs = tf.Variable(epochs)

  @tf.function(
    input_signature=[
      tf.TensorSpec(shape=[], dtype=tf.string),
      tf.TensorSpec(shape=[], dtype=tf.string),
    ]
  )
  def __call__(self, sentence, target_sentence, training=False):
    (result, tokens) = self.cgenerator(
      sentence, target_sentence, max_length=Max_Tokens, training=training
    )
    return result

  @tf.function
  def params(self):
    return {"loss": self.loss, "acc": self.acc, "epochs": self.epochs}

  def __str__(self):
    import json

    return json.dumps(
      {
        "loss": self.loss.numpy().tolist(),
        "acc": self.acc.numpy().tolist(),
        "epochs": self.epochs.numpy().tolist(),
      }
    )


# - routines ---
def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
  )
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match) / tf.reduce_sum(mask)
