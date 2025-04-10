import logging
import os
from pathlib import Path
from typing import Generator

from ...utils.gcs import download_file, exists, upload_prefix
from .hnn import HNN
from .sims_vectors import get_vectors

Local_Dir = "/tmp"
GCS_Dir = "gs://cogs_ds/hnns/"
Default_Name = "test_1000000"


class Sims:
  def __init__(self, name=Default_Name):
    self.name = name
    self.local_path = f"{Local_Dir}/{name}_sims_uris.txt"
    self.gcs_path = f"{GCS_Dir}/{name}_sims_uris.txt"
    self.hnn = HNN(name)
    self.uris = []
    self.load()

  def save(self):
    if not Path(self.local_path).exists():
      logging.info(f"Saving to {self.local_path}")
      with open(self.local_path, "w") as fh:
        fh.write("\n".join(self.uris))
      upload_prefix(self.local_path, self.gcs_path)

  def load(self):
    lp = Path(self.local_path)
    if exists(self.gcs_path) and not lp.exists():
      download_file(self.gcs_path, self.local_path)
    if lp.exists():
      with open(self.local_path, "r") as fh:
        self.uris = [li.strip() for li in fh]
    logging.info(f"Loaded uris from {self.local_path}")

  def _pfit(self, uris):
    self.uris = []
    for i, batch in enumerate(get_vectors(list(uris))):
      self.uris = self.uris + list(batch.keys())
      self.hnn.partial_fit(list(batch.values()))
      print("Sims batch", i)

  def fit(self, uri_gs: Generator):
    from more_itertools import chunked

    for batch in chunked(uri_gs, 100000):
      self._pfit(batch)

    self.hnn.save()
    self.save()

  def to_vectors(self, uris):
    ds = get_vectors(uris)  # generator of dicts (corresponding to batches)
    return dict((u, v) for d in ds for u, v in d.items())

  # Note: uris may be repeated, so result can't be a dict
  def from_vectors(self, vectors):
    labels, distances = self.hnn.lookup(vectors)
    distances = [d[0] for d in distances]
    ls = [li[0] for li in labels]
    uris = [self.uris[li] for li in ls]
    uds = zip(uris, distances)
    d = [(u, d) for u, d in uds]
    return d


# -------
def dot_product(v1: list[float], v2: list[float]) -> float:
  dp = sum(a * b for a, b in zip(v1, v2))
  return dp


# - ----------------------------------------------------
Sims_Dir = os.environ["HOME"] + "/data/sims/track/"
Sims_File = os.environ["HOME"] + "/data/sims/track/000000000000.avro"


def download_track_info(pname: str = Sims_Dir):
  pass


def gen_track_info(pname: str = Sims_Dir, field: str = "track_uri"):
  from pathlib import Path
  from random import shuffle

  from fastavro import reader

  paths = list(Path(pname).glob("*.avro"))
  shuffle(paths)

  for fp in paths:
    print("Reading: ", fp.name)
    with open(fp, "rb") as fh:
      for r in reader(fh):
        yield r[field]
  return


def test_create(n: int):
  import itertools

  name = f"test_{n}"
  s = Sims(name)
  turis = itertools.islice(gen_track_info(field="track_uri"), n)
  s.fit(turis)


def test_read():
  import itertools

  turis = list(itertools.islice(gen_track_info(field="track_uri"), 3))
  vs = list(itertools.islice(gen_track_info(field="vector"), 3))
  s = Sims()
  print("Orig", turis)
  print("HNN", s.from_vectors(vs))


if __name__ == "__main__":
  # test_create(1000000)
  test_read()
