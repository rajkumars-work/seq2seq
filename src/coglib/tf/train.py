import logging
import sys

import tensorflow as tf

from ..utils.text_io import upload_dicts_to_bq
from .data import TData
from .utils import (
  cp_callback,
  create_model,
  es_callback,
  get_strategy,
  load_model,
  make_tokenizers,
  save_model,
)


# ----------------
class Trainer:
  def __init__(
    self,
    model: tf.keras.Model,
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    name: str,
  ) -> None:
    self.model = model
    self.translator = None
    self.train_data = train_data
    self.val_data = val_data
    self.name = name
    self.save_every = 10
    self.start_epoch = 0
    self.es_callback = es_callback(name)
    self.cp_callback = cp_callback(name)
    self.loss = 0
    self.acc = 0
    self.epochs = 0

  def train(self, epochs):
    cbs = [self.es_callback, self.cp_callback]
    h = self.model.fit(self.train_data, epochs=epochs, callbacks=cbs).history
    self.loss = h["loss"][-1]
    self.acc = h["masked_accuracy"][-1]
    self.epochs = self.epochs + epochs
    print(f"Trained {epochs} /total {self.epochs} loss/acc: {self.loss}/{self.acc}")

  def save(self, td):
    from .models import CGenerator, ExportCGenerator

    tzs = make_tokenizers(td.pr.ct, td.pr.tt)
    gen = CGenerator(tzs, self.model)
    gen = ExportCGenerator(gen, self.loss, self.acc, self.epochs)
    self.translator = gen
    save_model(self.name, gen)

  def load(self):
    if not self.translator:
      self.translator = load_model(self.name)

  def translate(self, context, td):
    return self.predict(context, "", td)

  def predict(self, context, target, td):
    gen = self.translator(context, target).numpy().decode("utf-8")
    out = td.pr.tm.decode_pieces(gen.split())
    return td.pr.isrc.from_isrc_str(out) if td.pr.is_uri_prep() else out

  def evaluate(self, td):
    from itertools import islice

    uri_prep = td.pr.is_uri_prep()
    it = islice(td.get_dss(), 10)
    res = [
      (
        self.predict(c, t, td),
        td.pr.cm.decode_pieces(c.split()),
        td.pr.tm.decode_pieces(t.split()),
      )
      for c, t in it
    ]
    res = [
      {
        "pred": r,
        "context": c,
        "target": td.pr.isrc.from_isrc_str(t) if uri_prep else t,
      }
      for r, c, t in res
    ]
    return res


def load_train_objs(name, data, load_cp=True, uri_prep=True):
  td = TData(name, data, uri_prep=uri_prep)
  cvs, tvs = td.get_vocab_sizes()
  model = create_model(
    name, cvs, tvs, load_cp
  )  # create model, optionally load checkpoints
  train_set, val_set = td.get_dls()
  return train_set, val_set, model, td


def train():
  args = sys.argv

  name = args[1] if len(args) > 1 else "playlist_2m"
  data = args[2] if len(args) > 2 else "ml-sketchbook.cogs_data.playlist_isrc_2m"
  epochs = int(float(args[3])) if len(args) > 3 else 2

  with get_strategy().scope():
    train_data, val_data, model, td = load_train_objs(name, data, uri_prep=False)
    trainer = Trainer(model, train_data, val_data, name)
    trainer.train(epochs)
    trainer.save(td)
    upload_dicts_to_bq(trainer.evaluate(td), name)


def test():
  name = "playlist_2m"
  data = "ml-sketchbook.cogs_data.playlist_isrc_2m"
  train_set, val_set, model, td = load_train_objs(name, data, uri_prep=False)
  trainer = Trainer(model, train_set, val_set, name)
  trainer.train()
  trainer.save(td)
  trainer.load()
  upload_dicts_to_bq(trainer.evaluate(td), name)


if __name__ == "__main__":
  logging.basicConfig(level=logging.WARNING)
  # test()
  train()  # needs to be present
