import itertools
import logging
import pickle
from pathlib import Path

import numpy as np
from more_itertools import chunked

from coglib.utils.quantize import Quantization

from ...utils.gcs import download_file, exists, upload_prefix
from ..vectors.sims import Sims, gen_track_info
from .dl import DL, Batch_Size, Default_Sequence_Length, Default_Target_Dim, error_pct

# non_zero_size is the number of nonzero indices we want
# basis_dimension is the dimension of the basis vectors
# see: https://www.di.ens.fr/~fbach/mairal_icml09.pdf

Default_Quantization_Bits = 4  # 16 bins
L_Delim = "#"
S_Delim = ":"

Quant_Batches = 128
Default_Name = "test_1000000"


class Gist:
  Prefix = "gs://cogs_models/gist/"

  def __init__(
    self,
    name: str = Default_Name,
    non_zero_size: int = Default_Sequence_Length,
    basis_dimension: int = Default_Target_Dim,
    quant_bits: int = Default_Quantization_Bits,
    overwrite=False,
  ):
    self.dll = DL(name, non_zero_size, basis_dimension, overwrite)
    self.basis_digits = len(str(basis_dimension))
    self.quant_bits = quant_bits
    self.quant_digits = len(str(2**quant_bits)) if quant_bits else 0
    self.path = Gist.Prefix + name + ".pkl"
    self.local_path = f"/tmp/{name}.pkl"

    self.sims = Sims()
    self.load()

  # need to finish fitting before quantization, so cache first Quant_Batches batches
  def fit(self, data):
    batches = chunked(data, Batch_Size)
    qdata = []
    for i, batch in enumerate(batches):
      batch = np.array(batch, dtype=np.float32)
      self.dll.partial_fit(batch)
      if i < Quant_Batches:
        qdata.append(batch)
    self.quant = self._quantize(qdata)
    self.save()

  def partial_fit(self, batch):
    self.dll.partial_fit(batch)

  def _quantize(self, batches):
    first_batch = batches[0]
    arr = self.dll.transform(first_batch).flatten()
    nz = arr[np.where(arr != 0)]
    q = Quantization(self.quant_bits, nz)
    for i, batch in enumerate(batches):
      arr = self.dll.transform(batch).flatten()
      nz = arr[np.where(arr != 0)]
      q.tune(nz)
    return q

  def bases(self):
    return self.dll.bases()

  # transform original vectors to vectors in new space (dim = basis_dimension)
  def transform(self, data):
    return self.dll.transform(data)

  # from transformed vectors (dim = basis_dimension) to vectors in original space
  def restore(self, data):
    return self.dll.restore(data)

  # - IO -----------
  def save(self):
    self.save_local()
    upload_prefix(self.local_path, self.path)

  def save_local(self):
    if self.dll.overwrite or (not Path(self.local_path).exists()):
      logging.info(f"Saving dl, quant to {self.local_path}")
      with open(self.local_path, "wb") as fh:
        pickle.dump(self.dll, fh)
        pickle.dump(self.quant, fh)

  def load(self):
    if exists(self.path) and (not Path(self.local_path).exists()):
      download_file(self.path, self.local_path)
    self.load_local()

  def load_local(self):
    lp = Path(self.local_path)
    if lp.exists():
      with open(self.local_path, "rb") as fh:
        self.dll = pickle.load(fh)
        self.quant = pickle.load(fh)
      logging.info(f"Loaded dl and quant from {self.local_path}")

  # gists <-> numpy arrays
  def _to_gist(self, labels, qlabels):
    def lstr(label, qlabel):
      return (
        str(label).zfill(self.basis_digits)
        + L_Delim
        + str(qlabel).zfill(self.quant_digits)
      )

    def rstr(srow):
      srow.sort()
      return S_Delim.join(srow)

    vf = np.vectorize(lstr)
    sa = vf(labels, qlabels)
    return np.apply_along_axis(rstr, 1, sa)

  def _from_gist(self, gists):
    def strl(lstr):
      return int(lstr.split(L_Delim)[0])

    vstrl = np.vectorize(strl)

    def strq(lstr):
      return float(lstr.split(L_Delim)[1])

    vstrq = np.vectorize(strq)
    lqas = [gist.split(S_Delim) for gist in gists]
    return vstrl(lqas), vstrq(lqas)

  # - api -----
  # given an input vector(s), return token_prepresentation
  # index1.qindex1:index2:qindex2:...:indexn:qindexn
  def vectors_to_gists(self, ivs) -> list[str]:
    ivs = np.array(ivs)
    ivs = ivs if ivs.ndim == 2 else [ivs]
    indices, coeffs = self.dll._to_indices(ivs)
    qlabels = self.quant.quantize(coeffs) if self.quant_bits else coeffs
    gists = self._to_gist(indices, qlabels)
    return gists

  # given list of base + quant index tuples, return reconstructed input vector (in original space)
  def gists_to_vectors(self, gists: list[str]):
    indices, qlabels = self._from_gist(gists)
    coeffs = self.quant.reconstruct(qlabels)
    recon = self.dll._from_indices(indices, coeffs)
    return recon

  def uris_to_gists(self, uris) -> list[str]:
    vectors = [v for u, v in self.sims.to_vectors(uris).items()]
    gists = self.vectors_to_gists(vectors)
    return dict((u, g) for u, g in zip(uris, gists))

  def gists_to_uris(self, gists: list[str]):
    vectors = self.gists_to_vectors(gists)
    uris = self.sims.from_vectors(vectors)
    return dict((g, u) for g, u in zip(gists, uris))

  def err_pct(self, vectors):
    return self.dll.err_pct(vectors)


# --- testing -------
def vstr(v):
  return ", ".join([str(round(x, 2)) for x in v[:5]])


def fit_gist(n: int = 1000000, bits=Default_Quantization_Bits):
  name = f"test_{n}_{bits}"
  print("Fitting: ", name)

  g = Gist(name, quant_bits=bits, overwrite=True)
  vectors = itertools.islice(gen_track_info(field="vector"), n)
  g.fit(vectors)
  print("Fitting done; testing")

  vls = list(itertools.islice(gen_track_info(field="vector"), 30))
  print("DL Err: ", g.err_pct(vls))
  gists = g.vectors_to_gists(vls)
  print(gists)
  recon = g.gists_to_vectors(gists)
  print("Err pct", error_pct(vls, recon))


def test_gists(name: str = Default_Name):
  uris = [
    "spotify:track:2HRqTpkrJO5ggZyyK6NPWz",
    "spotify:track:2tHwzyyOLoWSFqYNjeVMzj",
  ]
  dl = Gist(name)
  gists = [g for u, g in dl.uris_to_gists(uris).items()]
  ruris = dl.gists_to_uris(gists)
  print(uris)
  print(gists)
  print(ruris)


if __name__ == "__main__":
  # logging.basicConfig(level=logging.DEBUG)
  logging.basicConfig(level=logging.INFO)
  n = 15000
  bits = 4
  name = f"test_{n}_{bits}"
  fit_gist(n=n, bits=bits)
  test_gists(name)
