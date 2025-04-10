import gzip
import json
import logging
from pathlib import Path
from typing import Dict

from .bq import download_as_json, upload_from_json
from .gcsdict import GcsDict

DEF_DIR = "/tmp/isrcs/"
IDB_PATH = "gs://cogs_models/data/isrcs.db"
UDB_PATH = "gs://cogs_models/data/uris.db"


idb = GcsDict(IDB_PATH)
udb = GcsDict(UDB_PATH)


def create_db():
  def _parse(li: str):
    d = json.loads(li)
    k = d["isrc"]
    v = (d["uri"], d["track"] + ":" + d["artist"])
    return k, v

  for f in Path(DEF_DIR).glob("*.json.gz"):
    logging.infop("File " + f)
    with gzip.open(f, "r") as fh:
      kvs = (_parse(li) for li in fh)
      ud = dict((u, i) for i, (u, n) in kvs)
      id = dict((i, (u, n)) for i, (u, n) in kvs)
      idb.dict().update(id)
      udb.dict().update(ud)
    udb.save()
    idb.save()


# to from isrc
def from_isrc(isrc: str, uri: bool = False) -> str:
  tup = idb.dict().get(isrc, (isrc, isrc))
  return tup[0] if uri else tup[1]


def to_isrc(uri: str) -> str:
  return udb.dict().get(uri, uri)


def from_isrcs(isrcs: str, uri: bool = False) -> str:
  return " ".join(from_isrc(isrc, uri) for isrc in isrcs.split())


def to_isrcs(uris: str) -> str:
  return " ".join(to_isrc(uri) for uri in uris.split())


# translators
def trans_dict(d: Dict[str, str], fr: bool = True, uri: bool = False) -> Dict[str, str]:
  return dict((k, from_isrcs(v, uri) if fr else to_isrcs(v)) for k, v in d.items())


tfile = "/tmp/tmpfile.json"


def trans_table(table: str, fr: bool = True, uri: bool = False) -> None:
  download_as_json(table, tfile, overwrite=True)
  ofile = trans_file(tfile, fr, uri)
  otable = f"{table}_e"
  upload_from_json(otable, ofile)
  return otable


otfile = "/tmp/otmpfile.json"


def trans_file(file: str, fr: bool = True, uri: bool = False) -> None:
  Path(otfile).unlink(True)
  with open(tfile, "r") as fh, open(otfile, "w") as ofh:
    for li in fh:
      d = json.loads(li)
      od = trans_dict(d, fr, uri)
      ofh.write(json.dumps(od) + "\n")
  return otfile


# ----
def test():
  table = "ml-sketchbook..playlists_20241023"
  table = "ml-sketchbook.tmp.playlists_1k"
  ot = trans_table(table)
  print(ot)


if __name__ == "__main__":
  res = create_db()
  print(res)
