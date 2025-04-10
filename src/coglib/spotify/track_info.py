import logging
from itertools import islice

from fastavro import reader

from ..utils.gcs import download_file, list_paths

Fields = ["context", "target"]
SIMS_PREFIX = "gs://sim-vectors-v4-0/track/sim-vectors.v4-0.track/"
TINFO_PREFIX = "gs://metadata-entities-extended/MetadataEntities.TrackExtended.gcs/"
Sims_Dir = "/tmp/"
Def_Date = "2025-01-15"


def gen_track_records(date: str = Def_Date, vectors: bool = False):
  prefix = SIMS_PREFIX if vectors else TINFO_PREFIX
  prefix = prefix + date
  paths = list_paths(prefix, ".avro")
  for path in paths:
    tname = Sims_Dir + path.split("/")[-1]
    download_file(path, tname)
    with open(tname, "rb") as fh:
      for d in reader(fh):
        yield d
  return


# - testing -------------------------------------------------------
def test():
  for r in islice(gen_track_records(vectors=True), 10):
    print(r)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  test()
