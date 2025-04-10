import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from .bq import upload_from_json
from .gcs import upload_prefix
from .sputils import sp_tokenizers
from .text_io import (
  ct_from_file,
  dir,
  inter_file,
  prep_file,
  prep_files,
  raw_files,
)


class DataPrep:
  def __init__(self, name: str, data: str):
    self.csp, self.tsp = sp_tokenizers(name)

  def encode_context(self, c):
    return " ".join(self.csp.encode_as_pieces(c))

  def decode_contexta(self, ca):
    return self.csp.decode_pieces(ca)

  def decode_contexts(self, cs):
    return self.csp.decode_pieces(cs.split())

  def encode(self, d):
    c = " ".join(self.csp.encode_as_pieces(d["context"]))
    t = " ".join(self.tsp.encode_as_pieces(d["target"]))
    return {"context": c, "target": t}

  def encode_target(self, t):
    return " ".join(self.tsp.encode_as_pieces(t))

  def decode_targeta(self, ta):
    return self.tsp.decode_pieces(ta)

  def decode_targets(self, ts):
    return self.tsp.decode_pieces(ts.split())

  def decode(self, ca: List[str], ta: List[str]):
    c = self.csp.decode_pieces(ca)
    t = self.tsp.decode_pieces(ta)
    return c, t

  def tokenizers(self):
    return self.csp, self.tsp

  def vocabs(self):
    cv = [self.csp.IdToPiece(i) for i in range(self.csp.vocab_size())]
    tv = [self.tsp.IdToPiece(i) for i in range(self.tsp.vocab_size())]
    return cv, tv


# - ------------------------------------
def processed_json_file(name) -> str:
  return dir(name) + "/processed.json"


def sp_prep(name: str, data: str) -> Tuple[List[str], List[str]]:
  dp = DataPrep(name, data)
  for rf in raw_files(name):
    inf = inter_file(rf)
    ifile = inf if Path(inf).exists() else rf
    pf = prep_file(rf)
    if not Path(pf).exists():
      with open(pf, "w") as ofh:
        for e in ct_from_file(ifile):
          d = dp.encode(e)
          ofh.write(json.dumps(d) + "\n")
  return dp.vocabs()


def upload_preprocessed_file(name: str, data: str) -> None:
  ofile = processed_json_file(name)
  if data.startswith("gs://"):
    path = f"gs://cogs_tmp/{name}.json"
    upload_prefix(ofile, path)
  else:
    path = f"bq://ml-sketchbook.cogs_tmp.{name}"
    upload_from_json(path, ofile)
  print(f"Uploaded to {path}")


def get_prep(name: str) -> Iterator[Dict[str, str]]:
  for file in prep_files(name):
    if Path(file).exists():
      with open(file, "r") as fh:
        for line in fh:
          yield json.loads(line)
  return


if __name__ == "__main__":
  from ..utils.sputils import create_models
  from ..utils.text_io import download

  name = "por_en"
  data = "ml-sketchbook.cogs_data.por_en"
  download(name, data)
  create_models(name, data)
  sp_prep(name, data)
