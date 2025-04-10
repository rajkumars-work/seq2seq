import logging
from pathlib import Path
from time import time

from fastavro import reader

from ..utils.gcsdict import GcsDict
from ..utils.tutils import memory

vector_dir = "/home/rkumar/data//sims/track"
vector_file = f"{vector_dir}/part-00000-of-00112.avro"
gcs_path = "gs://factoid_tmp/sims/mp_test"
# gcs_path = "gs://factoid_tmp/sims/small_test"

st = time()
mydb = GcsDict(gcs_path)
print("Loading took ", time() - st)


def load_vector_dir(dir: str):
  fs = Path(dir).glob("*.avro")
  for f in fs:
    load_vector_file(f)


def load_vector_file(f: str):
  st = time()
  with open(f, "rb") as fh:
    for r in reader(fh):
      mydb.dict()[r["uri"]] = r["vector"]
  rt = time() - st
  print("Read", f, rt, flush=True)
  mydb.commit()
  print("Commit", time() - st, flush=True)


def lookup_vector_file(f: str):
  st = time()
  with open(f, "rb") as fh:
    uris = [r["uri"] for r in reader(fh)]
  ts = (time() - st) + "secs" + memory()
  logging.log("reading took " + ts)
  st = time()
  for uri in uris:
    _ = mydb.dict().get(uri, "")
  ts = (time() - st) + "secs" + memory()
  logging.log("lookup took " + ts)


if __name__ == "__main__":
  lookup_vector_file(vector_file)
