import logging
import numbers

import numpy as np

# Quantization utilities
# quantize an array of floats into buckets (size specified by number of bits)
# uses lloyd-max for bucketization https://en.wikipedia.org/wiki/Lloyd%27s_algorithm

Default_Iter = 2
Default_Err_Ratio = 0.01
Max_Iter = 100
Max_Err_Ratio = 0.001


# bits = 0 =>no quantization
class Quantization:
  def __init__(self, bits: int, data, iterations=Max_Iter, max_err=Max_Err_Ratio):
    data = np.array(data, dtype=np.float32)
    self.levels = pow(2, bits)
    tolerance = (np.max(data) - np.min(data)) * max_err

    # start with centroids spaced by quantiles
    pm = np.arange(0, 1, 1.0 / self.levels)
    self.centroids = np.quantile(data, pm)
    self.boundaries = np.zeros(self.levels - 1)

    self._lloyd_max(data, iterations, tolerance)

  def tune(self, data, iterations=Default_Iter, max_err=Default_Err_Ratio):
    self._lloyd_max(data, iterations, max_err)

  def _lloyd_max(self, data, iterations, tolerance):
    if self.levels <= 1:
      return
    for it in range(iterations):
      # Update decision boundaries (midpoints of neighboring centroids)
      for i in range(len(self.centroids) - 1):
        self.boundaries[i] = (self.centroids[i] + self.centroids[i + 1]) / 2

      # Assign data points to the nearest centroid
      quantized_values = np.zeros_like(data)
      for i, x in enumerate(data):
        distances = np.abs(self.centroids - x)
        quantized_values[i] = self.centroids[np.argmin(distances)]

      # Update centroids (mean of the assigned data points)
      new_centroids = np.zeros_like(self.centroids)
      for i in range(len(self.centroids)):
        x = data > (self.boundaries[i - 1] if i > 0 else -np.inf)
        y = data <= (self.boundaries[i] if i < len(self.boundaries) else np.inf)
        index = x & y
        assigned_pts = data[index]
        new_centroids[i] = (
          np.mean(assigned_pts) if len(assigned_pts) > 0 else self.centroids[i]
        )
        logging.debug(f"{i} Assigned Pts {assigned_pts}")

      # Break if convergence is enough
      diff = new_centroids - self.centroids
      logging.debug(f"Old {self.centroids}")
      logging.debug(f"New {new_centroids}")
      max_diff = np.max(np.abs(diff))
      logging.debug(f"{i} Max Diff {max_diff} Tolerance {tolerance}")
      if max_diff < tolerance:
        logging.debug(f"Done after {it} iterations of {iterations}")
        break

      self.centroids = new_centroids

    min_c = np.min(self.centroids)
    max_c = np.max(self.centroids)
    logging.info(
      f"Its: {it}, max diff: {max_diff:.3f}, min: {min_c:.3f}, max: {max_c:.3f}"
    )

  def centroids(self, data):
    return self.centroids

  def quantize(self, darray):
    def closest_index(x) -> int:
      distances = np.abs(self.centroids - x)
      index = np.argmin(distances)
      # centroid = self.centroids[index]
      return index

    if self.levels <= 1:
      return darray
    data = np.array(darray, dtype=np.float32)
    vf = np.vectorize(closest_index)
    return vf(data)

  def reconstruct(self, indices):
    return (
      self.centroids[indices]
      if issubclass(indices.dtype.type, numbers.Integral)
      else indices
    )


# - usage example ------
if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  # logging.basicConfig(level=logging.DEBUG)

  data = [1.7, 0.11, 4.5, 1.2, 3.5, -0.8, 11.0, 7.0, 2.0]
  # data = np.array(data, dtype=np.float32)
  bits = 4
  print("Orig: ", data, "bits: ", bits)

  # initialize quantizer on some data
  q = Quantization(bits, data)

  # test the quantizer
  # tdata = np.array([2.11, 4.1, 1.1, -0.1])
  tdata = np.array([[2.11, 4.1, 1.1, -0.1], [3.4, -1.1, 0.1, 2.3]])
  print("Data: ", tdata)
  indices = q.quantize(tdata)
  res = q.reconstruct(indices)
  print("Ind: ", indices)
  print("LM: ", res)

  zq = Quantization(0, data)
  indices = zq.quantize(tdata)
  res = zq.reconstruct(indices)
  print("ZInd: ", indices)
  print("ZLM: ", res)
