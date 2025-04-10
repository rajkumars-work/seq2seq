# pylint: disable=import-not-at-top
import logging
import os
from typing import Any, List

import tensorflow as tf

from ..params import Default_Model_Params, Max_Vocab
from ..utils.gcs import get_dirs, gfs
from ..utils.prep import Prep

# - keras3 adjustments ------
try:
  import tensorflow.keras as keras

  # import tf_keras as keras # (causes errors in 2.14, 2.15)

  os.environ["TF_USE_LEGACY_KERAS"] = "1"
  keras3 = False

except ImportError:
  import keras

  keras3 = hasattr(keras, "version")


def is_keras3() -> bool:
  return keras3


logging.info(f"\nUsing keras3 ? {keras3}\n")

Prefix = "gs://cogs_models/tf/"


# - api --
def tf_models() -> List[str]:
  return get_dirs(Prefix, "model/")


def model_info(name):
  m = load_model(name)
  return (
    {
      "loss": m.loss.numpy().item(),
      "acc": m.acc.numpy().item(),
      "epoch": m.epochs.numpy().item(),
    }
    if (m and hasattr(m, "loss"))
    else {}
  )


# - model IO  -------
def cpath(name):
  prefix = f"{Prefix}{name}/"
  return prefix + "model.keras" if is_keras3() else prefix + "ck.ckpt"


def mpath(name):
  prefix = f"{Prefix}{name}/"
  return prefix + "model/model.keras" if is_keras3() else prefix + "model/"


def save_model(name, model):
  path = mpath(name)
  logging.info(f"Saving model to {path}")
  tf.saved_model.save(model, export_dir=path)


def load_model(name):
  path = mpath(name)
  return tf.saved_model.load(path) if gfs().exists(path) else None


# - callbacks -
# Can't currently save weights only in keras3 to gcs
# save whole model (and load with compie=False)
def save_weights_only() -> bool:
  return False if is_keras3() else True


def cp_callback(name: str) -> keras.callbacks.ModelCheckpoint:
  return keras.callbacks.ModelCheckpoint(
    filepath=cpath(name), verbose=1, save_weights_only=save_weights_only()
  )


def es_callback(name: str) -> keras.callbacks.EarlyStopping:
  return keras.callbacks.EarlyStopping(
    monitor="loss", patience=5, restore_best_weights=True
  )


def cp_exists(name: str) -> Any:
  path = cpath(name) if is_keras3() else cpath(name) + ".index"
  return gfs().exists(path)


def create_model(
  name, cvs: int = 0, tvs: int = 0, load_params: bool = True, keras3: bool = False
):
  if keras3:
    from .keras_models import (
      CustomSchedule,
      Transformer,
      masked_accuracy,
      masked_loss,
    )
  else:
    from .models import CustomSchedule, Transformer, masked_accuracy, masked_loss

  logging.info(f"Creating model model params: {save_weights_only()}")
  model_params = Default_Model_Params
  model_params["input_vocab_size"] = cvs if cvs else Max_Vocab
  model_params["target_vocab_size"] = tvs if tvs else Max_Vocab

  model = Transformer(**model_params)
  if cp_exists(name) and load_params:
    path = cpath(name)
    logging.info(f"Loading checkpoint from {path}; {save_weights_only()}")
    model.load_weights(path) if save_weights_only() else keras.models.load_model(path)
  learning_rate = CustomSchedule(model_params["d_model"])
  optimizer = keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
  )
  model.compile(
    loss=masked_loss, optimizer=optimizer, metrics=[masked_loss, masked_accuracy]
  )
  return model


# - preds --e
def predict(context: str, target: str, model, p: Prep) -> str:
  c = p.encode_context(context)
  t = p.encode_target(target)
  gen = model(c, t).numpy().decode("utf-8")
  out = p.tm.decode_pieces(gen.split())
  return p.isrc.from_isrc_str(out) if p.uri_prep else out


def translate(context: str, model, p: Prep) -> str:
  return predict(context, "", model, p)


# - parallelism ---
def get_strategy() -> Any:
  import tensorflow as tf
  from tensorflow.python.client import device_lib

  strategy = tf.distribute.MultiWorkerMirroredStrategy()
  devices = [x.device_type for x in device_lib.list_local_devices()]
  ds = "\nDevices: " + ", ".join(devices) + "\n"
  logging.info(ds)
  tpu = True if "TPU" in devices else False
  if tpu:
    os.environ["TPU_NAME"] = "local"
    os.environ["NEXT_PLUGGABLE_DEVICE_USE_C_API"] = "true"
    os.environ["TF_PLUGGABLE_DEVICE_LIBRARY_PATH"] = "/lib/libtpu.so"
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    ostr = "\nTPU all devices: " + ", ".join(
      [d.name for d in tf.config.list_logical_devices("TPU")]
    )
    logging.info(ostr)
  return strategy


# container object for tokenizers
def make_tokenizers(ct, tt):
  if is_keras3():
    import keras

    tokenizers = keras.Model()
  else:
    tokenizers = tf.Module()

  tokenizers.context = ct
  tokenizers.target = tt
  return tokenizers


if __name__ == "__main__":
  logging.basicConfig(level=logging.WARNING)
  logging.info(f"\n Using keras3: {keras3}")
  name = "por_en_tf_local"
  print(model_info(name))
