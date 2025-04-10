import logging
import os
import pickle
from pathlib import Path

import hnswlib
from fastavro import reader

from ...utils.gcs import download_file, exists, upload_prefix

VDim = 80
Sims_Dir = os.environ["HOME"] + "/data/sims/track"
Sims_File = Sims_Dir + "/000000000012.avro"
Max_Elements = 1000000

Local_Dir = "/tmp"
GCS_Dir = "gs://cogs_ds/hnns/"


class HNN:
  def __init__(self, name: str):
    self.local_path = f"{Local_Dir}/{name}_hnn.pkl"
    self.gcs_path = f"{GCS_Dir}/{name}_hnn.pkl"
    self.p = hnswlib.Index(space="l2", dim=VDim)
    self.p.init_index(max_elements=Max_Elements, ef_construction=10, M=16)
    self.p.set_ef(4)
    self.load()

  def save(self):
    if not Path(self.local_path).exists():
      logging.info(f"Saving to {self.local_path}")
      with open(self.local_path, "wb") as fh:
        pickle.dump(self.p, fh)
      upload_prefix(self.local_path, self.gcs_path)

  def load(self):
    lp = Path(self.local_path)
    if exists(self.gcs_path) and not lp.exists():
      download_file(self.gcs_path, self.local_path)
    if lp.exists():
      with open(self.local_path, "rb") as fh:
        self.p = pickle.load(fh)
    logging.info(f"Loaded hnn from {self.local_path}")

  def partial_fit(self, vectors):
    self.p.add_items(vectors, range(len(vectors)))

  def fit(self, vectors):
    self.partial_fit(vectors)
    self.save_local()

  def lookup(self, data):
    labels, distances = self.p.knn_query(data, k=1)
    return labels, distances

  def get_vectors(fn: str):
    with open(fn, "rb") as fh:
      return [r["vector"] for r in reader(fh)]


# - testing ---
def test_hnn(fit: bool = False):
  nn = HNN("test")
  vectors = HNN.get_vectors(Sims_File)[:10000]
  nn.fit(vectors)
  ls, ds = nn.lookup(vectors[10:20])
  print(ls, ds)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  test_hnn()
