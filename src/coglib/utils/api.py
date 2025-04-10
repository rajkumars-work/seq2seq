import logging
import os
from typing import Dict, List

import tensorflow as tf

from ..tf.utils import load_model, tf_models
from ..tf.utils import predict as tf_pred
from ..torch.utils import list_models, load, pred
from ..torch.utils import load_model as torch_load
from .cloud_ai import train_model
from .prep import Prep

tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
DEF_EPOCHS = 30
MAX_EPOCHS = 300

logging.getLogger().setLevel(logging.ERROR)


# - api -
def list() -> List[str]:
  return list_models() + tf_models()


def model_info(n: str) -> str:
  import json

  return json.dumps(_model_info(n))


def train(
  n: str, d: str, epochs: int = DEF_EPOCHS, torch: bool = True, uri_prep: bool = True
) -> Dict:
  version = "gpus" if torch else "tpu"
  epochs = min(max(epochs, 1), MAX_EPOCHS)
  j, m = train_model(n, d, epochs, version, uri_prep)
  logging.info("Cogs api: Job started")
  logging.info(j)
  return j


def predict(n: str, c: str, t: str) -> str:
  d = model_and_prep(n)
  mt = d.get("type", None)
  m = d.get("model", None)
  p = d.get("prep", None)
  if mt is None:
    return f"Model {n} unknown"
  return pred(c, t, m, p) if mt else tf_pred(c, t, m, p)


def translate(m: str, c: str) -> str:
  return predict(m, c, "")


def status(m: str) -> str: ...


# - helpers ---
def model(name):
  mt = is_torch_model(name)
  if mt is None:
    return mt
  if mt:
    return load(name)
  else:
    return load_model(name)


def _model_info(n: str) -> dict:
  d = {"name": n}
  mt = is_torch_model(n)
  if mt is None:
    return d
  m = model(n)

  if mt:  # torch
    d.update(
      {
        "type": "torch",
        "loss": round(m.get("loss"), 2),
        "acc": round(m.get("acc"), 2),
        "epochs": m.get("epochs"),
      }
    )
    return d
  else:  # tf
    d.update({"type": "tf"})
    if hasattr(m, "loss"):
      d.update(
        {
          "loss": round(m.loss.numpy().item(), 2),
          "acc": round(m.acc.numpy().item(), 2),
          "epochs": m.epochs.numpy().item(),
        }
      )
    return d


def model_and_prep(n):
  mt = is_torch_model(n)
  if mt is None:
    return {"model": None, "prep": None, "type": mt}
  if mt:
    return {"model": torch_load(n), "prep": Prep(n), "type": mt}
  else:
    return {"model": load_model(n), "prep": Prep(n, torch=False), "type": mt}


# torch => True, tf => False, not-found => None
def is_torch_model(name):
  tf_ms = tf_models()
  torch_ms = list_models()
  return True if name in torch_ms else False if name in tf_ms else None


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  d = "ml-sketchbook.cogs_data.por_en"
  ms = ["por_en_tf"]
  c = "o mundo e um ovo"
  print(list())
  for m in list():
    print(model_info(m))
    print(predict(m, c, ""))
