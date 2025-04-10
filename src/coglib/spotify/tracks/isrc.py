import json

from ..isrcs import Isrcs
from .track_metadata import get_isrcs

Default_Date = "2024-12-31"
Fields = ["context", "target", 'pred']


class Isrc:
  def __init__(self, date=Default_Date):
    self.isrcs = Isrcs(date)
    print("Isrc init done")

  # list of isrcs, returns dict with "key" on miss
  def to_isrcs(self, uris):
    return dict((k, v.lower() if v else k) for k, v in get_isrcs(uris).items())

  def to_isrc_str(self, uris):
    uris = uris.strip().split()
    isrcs = [(v.lower() if v else k) for k, v in get_isrcs(uris).items()]
    return " ".join(isrcs)

  def from_isrcs(self, isrcs):
    isrcs = (isrc.lower() for isrc in isrcs)
    urid = dict((k, v if v else k) for k, v in self.isrcs.uris(isrcs).items())
    return urid

  def from_isrc_str(self, isrcs):
    isrcs = isrcs.lower().strip().split()
    uris = [(v if v else k) for k, v in self.isrcs.uris(isrcs).items()]
    return " ".join(uris)

  # apply to fields in a dict
  def to_isrcd(self, d):
      return dict((k, self.to_isrc_str(v) if k in Fields else v) for k, v in d.items())

  def from_isrcsd(self, d):
    return dict((k, self.from_isrc_str(v) if k in Fields else v) for k, v in d.items())

  # apply to json file
  def to_isrc_json(self, jf: str, ofile: str):
    with open(jf, "r") as fh, open(ofile, "w") as ofh:
      for line in fh:
        od = self.to_isrcd(json.loads(line))
        ofh.write(json.dumps(od) + "\n")

  def from_isrc_json(self, jf: str, ofile: str):
    with open(jf, "r") as fh, open(ofile, "w") as ofh:
      for line in fh:
        od = self.from_isrcd(json.loads(line))
        ofh.write(json.dumps(od) + "\n")


# - ----------------------------------------------------
def test():
  i = Isrc()
  uris = [
    "spotify:track:7Dd4ONH9UK6P3QTPVlAmIh",
    "spotify:track:3meBo3B9V66M7y8QDhdytz",
    "spotify:track:58rPuuJooYn8llxG2Is8rL",
  ]
  d = {"DENI12100211": "spotify:track:7Dd4ONH9UK6P3QTPVlAmIh"}
  print(uris, d)

  iss = i.to_isrcs(uris)
  iss = [i for u, i in iss.items()]
  nuris = i.from_isrcs(iss)

  print(iss)
  print(nuris)


def test_file():
  i = Isrc()
  s = "/home/rkumar/cogs/playlists_uri/raw_0.jsons"
  t = "/home/rkumar/cogs/playlists_uri/foo_0.jsons"
  i.to_isrc_json(s, t)


if __name__ == "__main__":
  test_file()
