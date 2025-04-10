import os
from glob import glob

from datasets import load_dataset

from .models import Cache_Dir


def file_paths(name):
  path = os.environ["HOME"] + "/cogs/" + name + "/inter_*.jsons"
  return glob(path)


def ds_path(name):
  return f"{Cache_Dir}ds/{name}"


def cogs_ds(name: str):
  fpaths = file_paths(name)
  ds = load_dataset("json", data_files=fpaths)
  ds.save_to_disk(ds_path(name))
  return ds


def load_ds(name):
  return load_dataset(ds_path(name), split="train")


if __name__ == "__main__":
  ds = cogs_ds("por_en")["train"]
  print(len(ds))
  print(ds[0])
