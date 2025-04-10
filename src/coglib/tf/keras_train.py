import itertools
import json
import logging
import pathlib
import sys

import keras

from ..utils.preprocess import DataPrep, get_processed
from ..utils.sputils import create_models
from ..utils.text_io import download, model_prefix, pred_file, upload
from .tf_ds import Batch_Size, device_count, get_batched_data
from .tf_vocab import tokenizers
from .utils import (
  Default_model_params,
  cp_callback,
  create_model,
  es_callback,
  get_strategy,
)

Eval_Freq = 10


# - Model IO ----
def save_model(model, name):
  model.save(model_prefix(name) + "model.keras")


def load_model(name):
  return keras.models.load_model(model_prefix(name) + "model.keras", compile=False)


def create_generator(transformer, tzs):
  from .keras_models import CGenerator, ExportCGenerator

  cgenerator = CGenerator(tzs, transformer)
  cgenerator = ExportCGenerator(cgenerator)
  return cgenerator


def train_model(name, data, epochs, model_params, const_length: bool = True):
  logging.info(f"Creating sp models for {name}")
  create_models(name, data)  # creates vocab models
  tzs = tokenizers(name, data)
  train_ds, eval_ds = get_batched_data(name, data, tzs, const_length)

  model_params["input_vocab_size"] = tzs.context.get_vocab_size().numpy()
  model_params["target_vocab_size"] = tzs.target.get_vocab_size().numpy()

  with get_strategy().scope():
    transformer = create_model(name)
    transformer.fit(
      train_ds,
      validation_data=eval_ds,
      epochs=epochs,
      callbacks=[cp_callback(name), es_callback(name)],
    )

  transformer.summary()
  save_model(transformer, name)
  transformer = load_model(name)

  generator = create_generator(transformer, tzs)
  return generator


# - Batch Predict --
def batch_predict(model, name, data):
  from ..utils.preprocess import sp_preprocess

  # in case data has not already been downloaded, following two lines
  download(name, data)
  sp_preprocess(name, data)
  dp = DataPrep(name, data)
  it = eval_preds(model, name, dp)
  file = pred_file(name)
  if not pathlib.Path(file).is_file():
    with open(file, "w") as fh:
      len = 0
      for d in it:
        fh.write(json.dumps(d) + "\n")
        len = len + 1
    logging.info(f"Created {file} of length: {len}")
  logging.info(f"Uploading {file}")
  upload(name)


# - Testing --
def eval_preds(model, name, dp, count=10):
  it = get_processed(name)
  it = itertools.islice(it, count) if count else it
  return [translate(e["context"], e["target"], model, dp) for e in it]


def translate(sentence, ground_truth, translator, dp):
  gen = translator(sentence, "")
  gen = gen.numpy().decode("utf-8")
  gen = dp.decode_targets(gen)
  sentence = dp.decode_contexts(sentence)
  ground_truth = dp.decode_targets(ground_truth)
  return {
    "input": sentence,
    "ground_truth": ground_truth,
    "generation": gen,
  }


# - Training ---
def train():
  args = sys.argv

  name = args[1] if len(args) > 1 else "por_en_keras"
  data = args[2] if len(args) > 2 else "ml-sketchbook.cogs_data.por_en"
  epochs = int(float(args[3])) if len(args) > 3 else 30

  model_params = Default_model_params
  const_length_batches = True
  # const_length_batches = True if tpu else False

  logging.info(f"Device Count: {device_count}, Batch Size: {Batch_Size}")
  logging.info(f"{name}:{data}\n\tepochs:{epochs}")
  logging.info(f"Using constant Batch Size: {const_length_batches}")

  logging.info(f"Training for {name} starting")
  model = train_model(name, data, epochs, model_params, const_length_batches)
  logging.info(f"Training for {name} done. Doing an evaluation")

  batch_predict(model, name, data)


def test():
  name = "por_en_keras"
  data = "ml-sketchbook.cogs_data.por_en"
  tzs = tokenizers(name, data)
  model = load_model(name)
  generator = create_generator(model, tzs)
  batch_predict(generator, name, data)


# - Entry Point ---
if __name__ == "__main__":
  logging.basicConfig(level=logging.WARNING)
  train()
  # test()
