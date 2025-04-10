import os
from itertools import islice
from time import time

import grpc
from more_itertools import chunked

from coglib.spotify.vectors.sims_pb2 import GetVectorsRequest
from coglib.spotify.vectors.sims_pb2_grpc import VectorsStub
from coglib.utils.dns_lookup import resolve
from coglib.utils.gcloud_auth import (
  default_creds,
  default_creds_from_gcs,
  get_impersonation_token,
  get_token,
)

Audience = "http://vector-service"
Sims_Service = "_spotify-vector-service._grpc.services.gew1.spotify.net"
SA = "flux-dev-caller@vector-serving-systems.iam.gserviceaccount.com"
Batch_Size = 10000

host = resolve(Sims_Service)


def get_vector(uri: str) -> list[float]:
  rmap = next(get_vectors([uri]))
  return rmap.get(uri, [])


# generator of dicts (each a batch)
def get_vectors(uris: list[str]) -> dict[str, list[float]]:
  batches = chunked(uris, Batch_Size)
  for batch in batches:
    yield _get_batch(batch)
  return


def _get_batch(uris: list[str]) -> list:
  request = GetVectorsRequest(model="v4-0", uris=uris)
  # creds = default_creds()
  creds = default_creds_from_gcs()
  auth_token = get_token(creds)
  im_token = get_impersonation_token(auth_token, SA)
  metadata = (("spotify-service-identity", f"{im_token}"),)
  with grpc.insecure_channel(host) as channel:
    stub = VectorsStub(channel)
    response, call = stub.GetVectors.with_call(
      request=request, metadata=metadata, timeout=60
    )
  return dict((rv.uri, rv.vector) for rv in response.vectors)


# - ----------------------------------------------------
Sims_Dir = os.environ["HOME"] + "/data/sims/track/"
Sims_File = os.environ["HOME"] + "/data/sims/track/000000000000.avro"


def gen_turis(fn: str = Sims_File):
  from pathlib import Path

  from fastavro import reader

  path = Path(fn)
  with open(path, "rb") as fh:
    for r in reader(fh):
      yield r["track_uri"]
  return


def test():
  count = 0
  st = time()
  uris = islice(gen_turis(), 2)
  # uris = gen_turis()
  vs = get_vectors(uris)
  for i, b in enumerate(vs):
    count = count + len(b)
    print(i, count)
  et = time() - st
  r = et * 1000 / count
  print(f"ET(secs): {et:.3f} at {r} ms/rec")


def rtest():
  uuris = [
    "spotify:userid:96d0682b21824724adc3d1ff91e14787",
    "spotify:userid:e8a56e3eb0ad4d698ebf835439212afd",
  ]
  uuri = uuris[0]
  uv = get_vector(uuri)
  print(uv)


if __name__ == "__main__":
  rtest()
