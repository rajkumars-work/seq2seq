from itertools import islice

import torch
import torch.nn as nn

from ..utils.gcs import gfs
from .s2s_data import SData
from .s2s_models import Seq2SeqTransformer, translate
from .utils import get_device

device = get_device()
DEVICE = device


def load_model(name, data):
  sd = SData(name, data)
  svl, tvl = sd.get_vocab_sizes()
  # svl = 12981; tvl = 12851
  print("Vocab Sizes: ", "source: ", svl, "target: ", tvl)
  model: torch.nn.Module = create_model(svl, tvl)
  model = model.to(device)
  loss = 0
  epochs = 0
  path = cpath(name)
  if gfs().exists(path):
    with gfs().open(path, "rb") as fh:
      d = torch.load(fh, map_location=DEVICE)
      model.load_state_dict(d["model_state_dict"])
      loss = d["loss"]
      epochs = d["epoch"]
      print("Loss: ", loss, "Epochs: ", epochs)
      return model, loss, epochs, sd


def create_model(ivs, tvs):
  params = {
    "d_model": 128,
    "dff": 512,
    "num_layers": 6,
    "num_heads": 8,
    "dropout_rate": 0.1,
    "input_vocab_size": ivs,
    "target_vocab_size": tvs,
  }
  model = Seq2SeqTransformer(**params)
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  return model


def load_data(sd):
  _, data = sd.get_dss()
  data = islice(data, 10)
  st, tt, _ = sd.get_transforms()
  res = [(translate(model, c, st, tt, DEVICE), t) for c, t in data]
  res = [(sd.dp.decode_targets(c), sd.dp.decode_targets(t)) for c, t in res]
  return res


def cpath(name):
  return f"gs://cogs_models/torch/{name}.pt"


if __name__ == "__main__":
  name = "por_en"
  data = "ml-sketchbook.cogs_data.por_en"

  model, loss, epochs, sd = load_model(name, data)
  res = load_data(sd)
  for c, t in res:
    print("\n", c)
    print(t)
  # upload(res, otable)
