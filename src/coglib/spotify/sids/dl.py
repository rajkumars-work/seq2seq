import logging
import pickle
from pathlib import Path

import numpy as np
from more_itertools import chunked
from sklearn.decomposition import MiniBatchDictionaryLearning

from ..track_info import gen_track_records
from ..vectors.sims import Sims

# non_zero_size is the number of nonzero indices we want
# basis_dimension is the dimension of the basis vectors
# see: https://www.di.ens.fr/~fbach/mairal_icml09.pdf

# Source_Dim = 80
Default_Target_Dim = 2048
Default_Sequence_Length = 8

Max_Iter = 10
Batch_Size = 4096


def error_pct(o, r):
  o = np.array(o)
  r = np.array(r)
  diff = np.sum(np.square(np.subtract(o, r))) / np.sum(np.square(o))
  return round(100 * diff, 1)


class DL:
  Prefix = "gs://cogs_models/dl/"

  def __init__(
    self,
    name: str,
    non_zero_size: int = Default_Sequence_Length,
    basis_dimension: int = Default_Target_Dim,
    overwrite=False,
    verbose=False,
  ):
    self.overwrite = overwrite
    self.basis_dim = basis_dimension
    self.basis_digits = len(str(basis_dimension))
    self.non_zero_size = non_zero_size
    self.path = DL.Prefix + name + ".pkl"
    self.local_path = f"/tmp/dl_{name}.pkl"
    self.dl = MiniBatchDictionaryLearning(
      n_components=basis_dimension,
      transform_n_nonzero_coefs=non_zero_size,
      # transform_algorithm="lars",  # or 'omp'
      transform_algorithm="omp",
      max_iter=Max_Iter,
      n_jobs=-1,
      random_state=42,
      verbose=verbose,
    )
    self.load_local()
    self.sims = Sims()

  def fit(self, X, save: bool = True, err: bool = True):
    X = np.array(list(X), dtype=np.float32)
    T = self.dl.fit(X)
    print("Fitting done")
    if err:
      T = self.dl.fit_transform(X)
      X_hat = T @ self.dl.components_
      mean_error = np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X**2, axis=1))
      print("MeN Error: ", mean_error)
    if save:
      self.save_local()

  def batch_fit(self, data, save: bool = True, err: bool = False):
    batches = chunked(data, Batch_Size)
    for i, batch in enumerate(batches):
      X = np.array(batch, dtype=np.float32)
      self.dl.partial_fit(X)
      logging.info(f"Batch {i}")

      if err:
        T = self.dl.transform(X)
        X_hat = self.restore(T)
        mean_error = np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X**2, axis=1))
        logging.info(f"Mean Err: {mean_error}")

    if save:
      self.save_local()

  def partial_fit(self, batch):
    self.dl.partial_fit(batch)

  def bases(self):
    return self.dl.components_

  # transform original vectors to vectors in new space (dim = basis_dimension)
  def transform(self, data):
    return self.dl.transform(data)

  # from transformed vectors (dim = basis_dimension) to vectors in original space
  def restore(self, data):
    return data @ self.dl.components_

  def err_pct(self, data):
    va = np.array(data, dtype=np.float32)
    transformed = self.dl.transform(va)
    recreated = self.restore(transformed)
    return error_pct(va, recreated)

  # - IO -----------
  def save_local(self):
    if self.overwrite or not Path(self.local_path).exists():
      with open(self.local_path, "wb") as fh:
        pickle.dump(self.dl, fh)

  def load_local(self):
    lp = Path(self.local_path)
    if lp.exists():
      with open(self.local_path, "rb") as fh:
        self.dl = pickle.load(fh)

  # - Utils ----------------
  def _to_indices(self, va):
    va = np.array(va, dtype=np.float32)
    ts = self.dl.transform(va)
    abs = np.abs(ts)
    indices = np.argsort(-abs, axis=-1)
    coeffs = np.take_along_axis(ts, indices, axis=-1)
    indices = np.where(coeffs == 0, 0, indices)
    indices = indices[:, : self.non_zero_size]
    coeffs = coeffs[:, : self.non_zero_size]
    return indices, coeffs

  def _from_indices(self, indices, coeffs):
    coeffs = np.array(coeffs, dtype=np.float32)
    coeffs = coeffs[:, :, np.newaxis]
    vectors = self.dl.components_[indices]
    vectors = vectors * coeffs
    recreated = vectors.sum(axis=1)
    return recreated

  # - api -----
  # given an input vector(s), return token_prepresentation
  def vectors_to_dls(self, ivs):
    ivs = np.array(ivs)
    ivs = ivs if ivs.ndim == 2 else [ivs]
    return self._to_indices(ivs)

  # given indices + coeffs, reconst input vector (in original space)
  def dls_to_vectors(self, indices, coeffs):
    return self._from_indices(indices, coeffs)

  def uris_to_dls(self, uris):
    vectors = [v for u, v in self.sims.to_vectors(uris).items()]
    indices, coeffs = self.vectors_to_dls(vectors)
    return indices, coeffs

  def dls_to_uris(self, indices, coeffs):
    vectors = self.dls_to_vectors(indices, coeffs)
    uris = self.sims.from_vectors(vectors)
    return uris

  def reconstruct_error(self, uris):
    vectors = [v for u, v in self.sims.to_vectors(uris).items()]
    indices, coeffs = self.vectors_to_dls(vectors)
    rvectors = self.dls_to_vectors(indices, coeffs)
    return error_pct(vectors, rvectors)


# --- testing -------
# fit n sims vectors, then compute reconstruction error
def fit_dl(n: int = 1000):
  import itertools

  name = f"test_{n}"
  dl = DL(name)
  records = itertools.islice(gen_track_records(vectors=True), n)
  vectors = (r["vector"] for r in records)
  dl.fit(vectors, err=True)


def test_dl(name: str):
  uris = [
    "spotify:track:2HRqTpkrJO5ggZyyK6NPWz",
    "spotify:track:2tHwzyyOLoWSFqYNjeVMzj",
  ]
  dl = DL(name)
  indices, coeffs = dl.uris_to_dls(uris)
  ruris = dl.dls_to_uris(indices, coeffs)
  print(uris)
  print(indices, coeffs)
  print(ruris)
  print("Err pct", dl.reconstruct_error(uris))


if __name__ == "__main__":
  # logging.basicConfig(level=logging.DEBUG)
  logging.basicConfig(level=logging.INFO)
  fit_dl()
