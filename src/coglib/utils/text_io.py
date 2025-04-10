import itertools
import json
import logging
import pathlib
import random
from collections import Counter
from itertools import chain
from typing import Any, Dict, Iterator, List

from more_itertools import chunked

from . import bq, gcs
from .bq import download_as_jsons, upload_from_json
from .gcs import download_file, upload_prefix

BASE_PREFIX = "gs://cogs_models/data/"
CP_PREFIX = "gs://cogs_logs/cogs/"
DS_PREFIX = "gs://cogs_ds/"

# Preprocessing (breaks down large table into files of size Max_Rows)
Raw_Prefix = "raw"
Inter_Prefix = "inter"
Prep_Prefix = "prep"
Max_Rows = 1000000


# Paths
def prefix(name: str = "") -> str:
  return BASE_PREFIX + name


def model_prefix(name: str) -> str:
  return BASE_PREFIX + name + "/model/"


def vocab_prefix(name: str) -> str:
  return BASE_PREFIX + name + "/vocab/"


def ds_path(name):
  return f"{DS_PREFIX}{name}/"


def vocab_path(name):
  return f"{DS_PREFIX}{name}/vocabs/"


def dir(name: str) -> str:
  HOME = str(pathlib.Path.home())
  dir = f"{HOME}/cogs/{name}"
  pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
  return dir


def check_prefix(name: str) -> str:
  return f"{CP_PREFIX}{name}/checkpoints/"


def check_path(name):
  path = dir(name) + "/checkpoints/"
  pathlib.Path(path).mkdir(parents=True, exist_ok=True)
  return path


# If parent is True, assumes path is a file else dir
def mkdir(path: str, parent=True) -> bool:
  dir = pathlib.Path(path).parent if parent else path
  print(dir)
  pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
  return pathlib.Path(dir).is_dir()


# gcs path
def base_prefix(name):
  return prefix(name)


# file io
def write_vocab_file(filepath, vocab):
  with open(filepath, "w") as f:
    for token in vocab:
      print(token, file=f)


def read_vocab_file(filepath):
  with open(filepath, "r") as fh:
    return [w.rstrip() for w in fh]


def exists(name):
  HOME = str(pathlib.Path.home())
  dir = f"{HOME}/cogs/{name}"
  return pathlib.Path(dir).exists()


def rows(name, data, count=10):
  return itertools.islice(get_json(name, data), count)


def pred_file(name):
  return dir(name) + "/pred.json"


Pred_Prefix = "bq://ml-sketchbook.cogs_evals"
Data_Prefix = "bq://ml-sketchbook.cogs_tmp"


def pred_path(name):
  return Pred_Prefix + "." + name


def vocab_files(name):
  d = dir(name)
  return f"{d}/context.vocab", f"{d}/target.vocab"


# - files --
def json_from_files(files):
  for file in files:
    with open(file, "r", encoding="utf-8") as fh:
      for line in fh:
        if line:
          try:
            yield json.loads(line)
          except Exception as err:
            logging.error(err)
  return


def json_from_file(file):
  with open(file, "r", encoding="utf-8") as fh:
    for line in fh:
      if line:
        try:
          yield json.loads(line)
        except Exception as err:
          logging.error(err)
  return


def split_dir(sdir: str, tdir: str, name: str, ftype: str, max_rows=Max_Rows):
  fhs = [open(file, "r") for file in pathlib.Path(sdir).glob("*.json")]
  its = chain(fhs)
  _split_it(its, tdir, name, ftype, max_rows)


def split_file(src: str, dir: str, max_rows=Max_Rows):
  sp = pathlib.Path(src)
  if sp.is_file():
    name, ftype = sp.name.split(".")
    it = open(src, "r")
    _split_it(it, dir, name, ftype, max_rows)


def _split_it(it: Iterator, dir: str, name: str, ftype: str, max_rows):
  for i, c in enumerate(chunked(it, max_rows)):
    fn = f"{dir}/{name}_{i}.{ftype}"
    if not pathlib.Path(fn).exists():
      with open(fn, "w") as ofh:
        for li in c:
          ofh.write(li)


# given a name, what are the raw downloaded files called
# Note: currently dows not work for files from gs:// just bq


def raw_files(name) -> List[str]:
  d = dir(name)
  ps = pathlib.Path(d).glob(f"{Raw_Prefix}_*.jsons")
  return [p.as_posix() for p in ps]


def prep_file(raw_file: str) -> str:
  return raw_file.replace(Raw_Prefix, Prep_Prefix)


def inter_file(raw_file: str) -> str:
  return raw_file.replace(Raw_Prefix, Inter_Prefix)


def prep_files(name: str) -> List[str]:
  return [prep_file(r) for r in raw_files(name)]


def prep_sizes(name) -> Dict[str, int]:
  ps = [prep_file(r) for r in raw_files(name)]
  return dict(sorted([(p, file_lines(p)) for p in ps]))


def file_lines(fn: str) -> int:
  with open(fn, "r") as fh:
    return sum(1 for li in fh)


def file_for_index(name, index: int) -> str:
  fi = index // Max_Rows
  d = dir(name)
  fn = f"{d}/{Prep_Prefix}_{fi}.jsons"
  return fn if pathlib.Path(fn).is_file() else ""


# --
def download(
  name,
  path,
):
  d = dir(name)
  tf = f"{d}/{Raw_Prefix}.jsons"
  pathlib.Path(d).mkdir(parents=True, exist_ok=True)
  if path.startswith("gs://"):
    download_file(path, tf)
    split_file(tf, d)
  else:
    download_as_jsons(path, d, Raw_Prefix, Max_Rows)


def upload(name, path=None, json_file=None):
  path = path if path else pred_path(name)
  json_file = json_file if json_file else pred_file(name)
  if path.startswith("gs://"):
    upload_prefix(json_file, path)
  else:
    upload_from_json(path, json_file)


def eval_bq_prefix(name):
  return "bq://ml-sketchbook.cogs_evals." + name


def upload_dicts_to_bq(dicts, name: str):
  fn = "/tmp/foo.json"
  with open(fn, "w") as fh:
    for d in dicts:
      fh.write(json.dumps(d) + "\n")
  upload_from_json(eval_bq_prefix(name), fn)


def get_json(name: str, data: str) -> Iterator[Dict[str, str]]:
  if data:
    download(name, data)
  return (
    {"context": e.get("context", ""), "target": e.get("target", "")}
    for e in json_from_files(raw_files(name))
  )


def ct_from_file(fn: str) -> Iterator[Dict[str, str]]:
  return (
    {"context": e.get("context", ""), "target": e.get("target", "")}
    for e in json_from_file(fn)
  )


# each batch is a tuple of string-lists
def get_batched_json(name: str, data: str, bs: int):
  cit = chunked(get_json(name, data), bs)
  return (
    ([e["context"] for e in c], [e["target"] for e in c]) for c in cit if len(c) == bs
  )


def shuffle_it(it: Iterator[Any], bs: int) -> Iterator[Any]:
  cit = chunked(it, bs)
  for c in cit:
    random.shuffle(c)
    for e in c:
      yield e


# data helpers
def path_ok(path):
  return gcs.path_ok(path) if str.startswith(path, "gs://") else bq.path_ok(path)


def format_ok(name, path):
  try:
    it = get_json(name, path)
    keys = next(it).keys()
    return "context" in keys and "target" in keys
  except Exception as e:
    logging.error(e)
    return False


def data_ok(name, path):
  if path_ok(path):
    if format_ok(name, path):
      for count, _ in enumerate(get_json(name, path)):
        pass
      mins_per_epoch = int((count * 128) / (64 * 60 * 1000))
      mins_per_epoch = max(1, mins_per_epoch) if count > 0 else mins_per_epoch
      return mins_per_epoch
    else:
      print("Could not read data from path", path, " with context and target fields")
      return 0
  else:
    print(
      f"Data path {path} should be of form gs://... or bq://<project-id>:dataset/table"
    )
    return 0


# - stats ---
def counts(es):
  cu = Counter()
  tu = Counter()
  for e in es:
    cws = e["context"].split() + ["rows"]
    tws = e["target"].split() + ["rows"]
    cu.update(cws)
    tu.update(tws)
  return cu, tu


def write_counts(name: str, data: str) -> None:
  es = get_json(name, data)
  cu, tu = counts(es)

  cfile = dir(name) + "/ccounts.tsv"
  tfile = dir(name) + "/tcounts.tsv"
  with open(cfile, "w") as cfh:
    for k, v in cu.most_common():
      cfh.write(f"{k}\t{v}\n")
  with open(tfile, "w") as tfh:
    for k, v in tu.most_common():
      tfh.write(f"{k}\t{v}\n")


# chars and words
def max_length(name, data) -> int:
  ml, mw = 0, 0
  ms = ""
  for e in get_json(name, data):
    c = e["context"]
    t = e["target"]
    cw = c.split()
    tw = t.split()
    n = max(len(c), len(t))
    w = max(len(cw), len(tw))
    ml = n if n > ml else ml
    mw = w if w > mw else mw
    ms = c if len(c) > len(ms) else ms
    ms = t if len(t) > len(ms) else ms
  return ml, mw, ms


if __name__ == "__main__":
  name = "por_en_test"
  data = "gs://cogs-data/datasets/lang/p_e_50m.json"
  download(name, data)
