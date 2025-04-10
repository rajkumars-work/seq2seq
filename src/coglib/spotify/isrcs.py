import logging
import os
import tempfile
from functools import partial
from itertools import islice
from multiprocessing import Pool, cpu_count

from fastavro import reader

from ..utils.gcs import download_file, list_paths
from ..utils.gcsdict import GcsDict

GCS_PREFIX = "gs://cogs_models/data/dbs/"
TINFO_PREFIX = "gs://metadata-entities-extended/MetadataEntities.TrackExtended.gcs/"
DEF_DATE = "2025-01-20"
DEF_SIZE = 25000000
ISRC_LEN = 12

# ISO-3901 (ISRC) code: CC-XXX-YY-NNNNN (w/o hyphens)
# CC - Countrycode, XXX - Registrant -code, YY - year, NNNNN - id
# https://isrc.ifpi.org/en/isrc-standard/structure


class Isrcs:
  def __init__(self, date=DEF_DATE):
    self.date = date
    ipath = GCS_PREFIX + f"isrcs_{self.date}"
    self.idb = GcsDict(ipath)
    logging.info("Loaded " + ipath)

  def uri(self, id):
    id = id.lower()
    return self.idb.dict().get(id) if len(id) == ISRC_LEN else id

  def uris(self, isrcs):
    d = dict((isrc.lower(), self.idb.dict().get(isrc.lower(), "")) for isrc in isrcs)
    return d

  # count limits to tracks with rank < count
  def update_db(self, count=DEF_SIZE, files=0):
    ipath = GCS_PREFIX + f"isrcs_{self.date}"
    self.idb = GcsDict(ipath, create=True)
    logging.info(f"Loading dbs for {self.date} for {count} isrcs")

    prefix = TINFO_PREFIX + self.date
    paths = list_paths(prefix, ".avro")
    paths = islice(paths, files) if files else paths

    p = partial(process_file, count=count)
    with Pool(cpu_count()) as po:
      ids = po.map(p, paths)
    for id in ids:
      self.idb.dict().update(id)
    self.idb.save()
    self.idb = GcsDict(ipath)


def process_file(path, count):
  def _parse(d):
    u = d["track_uri"]
    i = d["track_isrc"].lower()
    p = d.get("global_popularity")
    r = p.get("rank", 0) if p else 0
    return {"uri": u, "isrc": i, "rank": r}

  fp, tname = tempfile.mkstemp()
  os.close(fp)
  logging.info("File " + path)
  download_file(path, tname, True)
  print("Tfile", tname)
  with open(tname, "rb") as fh:
    ds = (_parse(d) for d in reader(fh))
    ds = (d for d in ds if d.get("rank"))
    ds = (d for d in ds if d.get("rank") < count) if count > 0 else ds
    id = dict((d["isrc"], d["uri"]) for d in ds)
    os.unlink(tname)
  return id


# ----
def test_read():
  ti = Isrcs()
  turis = [
    "spotify:track:7Dd4ONH9UK6P3QTPVlAmIh",
    "spotify:track:3meBo3B9V66M7y8QDhdytz",
    "spotify:track:58rPuuJooYn8llxG2Is8rL",
  ]

  isrcs = ["FR6V80077945", "DK4V61900702", "DENI12100211"]
  print(turis)
  uris = ti.uris(isrcs)
  print(uris)


def test_create():
  ti = Isrcs()
  ti.update_db()


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  test_read()
