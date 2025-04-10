import json
import logging
import shutil
from pathlib import Path

from ..utils.gcs import gfs
from ..utils.preprocess import sp_prep
from ..utils.sputils import create_models, sp_tokenizers
from ..utils.text_io import (
  dir,
  download,
  inter_file,
  prep_file,
  raw_files,
  write_vocab_file,
)

# - Prepare data --------
# raw text
# intermediate text (after applying uri to isrc conversions for ex.)
# preprocessed text (apply subword tokenization)
# context and target vocabularies and tokenizers


class Prep:
  def __init__(
    self, name: str, data: str = None, torch: bool = True, uri_prep: bool = False
  ):
    self.name = name
    self.data = data
    self.torch = torch
    self.uri_prep = uri_prep
    self.base_prefix = f"gs://cogs_models/data/{name}"
    self.base_dir = dir(name)
    self.files = {
      "vocab": ["context_vocab.txt", "target_vocab.txt"],
      "raw": ["raw_0.jsons"],
      "inter": ["inter_0.jsons"],
      "prep": ["prep_0.jsons"],
    }
    self.prepare_data(data)

  def prepare_data(self, data):
    logging.info(f"Prep: creating data for {self.name}, {self.data}")
    if self.data_exists(False):
      logging.info("Data exist remotely")
      self.update_files(local=False)
      self.download()

    if (not self.data_exists()) and data:
      logging.info("Data does not exist locally")
      self.create_data(data)
    else:
      logging.info("Data exist locally")

    logging.info("Checking if models exist")
    self.create_models(data)

    self.update_files()
    self.cm, self.tm = sp_tokenizers(self.name)
    self.create_tokenizers()

    if not self.data_exists(False):
      logging.info("Data does not exist remotely. Uploading")
      self.upload()

    logging.info(f"Data prep done for {self.name}, {data}")

  def data_exists(self, local: bool = True) -> bool:
    return all([self.files_exist(v, local) for k, v in self.files.items()])

  def is_uri_prep(self, local: bool = True):
    if self.data_exists(local):
      r = self.files["raw"][0]
      rsize = (
        Path(self.local_file(r)).stat().st_size
        if local
        else gfs().size(self.remote_path(r))
      )

      i = self.files["inter"][0]
      isize = (
        Path(self.local_file(i)).stat().st_size
        if local
        else gfs().size(self.remote_path(i))
      )
      logging.info(f"Inter size {isize},  Raw size {rsize}")
      return isize != rsize
    else:
      return self.uri_prep

  def create_data(self, data):
    download(self.name, data)
    self.create_intermediate_files()

  def create_models(self, data):
    create_models(self.name, data)
    self.generate_vocabs()
    self.update_files()
    self.upload()

  def to_isrc_json(self, rf, inf):
    with open(inf, "r") as fh, open(rf, "w") as ofh:
      for line in fh:
        od = self.to_isrc(json.loads(line))
        ofh.write(json.dumps(od) + "\n")

  def create_intermediate_files(self):
    if not self.file_exists(self.files["inter"], True):
      if self.uri_prep:
        from ..spotify.tracks.isrc import Isrc

        self.isrc = Isrc()
        for rf in raw_files(self.name):
          inf = inter_file(rf)
          if not Path(inf).exists():
            self.isrc.to_isrc_json(rf, inf)
      else:
        for rf in raw_files(self.name):
          inf = inter_file(rf)
          shutil.copyfile(rf, inf)

  def update_files(self, local=True):
    if local:
      self.files["raw"] = [Path(f).name for f in raw_files(self.name)]
    else:
      paths = gfs().glob(self.base_prefix + "/raw_*")
      self.files["raw"] = [p.split("/")[-1] for p in paths]
    self.files["inter"] = [inter_file(f) for f in self.files["raw"]]
    self.files["prep"] = [prep_file(f) for f in self.files["inter"]]

  def local_file(self, file):
    return f"{self.base_dir}/{file}"

  def remote_path(self, file):
    return f"{self.base_prefix}/{file}"

  def files_exist(self, files, local: bool = True) -> bool:
    es = [self.file_exists(file, local) for file in files]
    return all(es)

  def file_exists(self, file, local: bool = True) -> bool:
    return (
      Path(self.local_file(file)).exists()
      if local
      else gfs().exists(self.remote_path(file))
    )

  def put(self, files):
    for file in files:
      if not self.file_exists(file, False):
        gfs().put_file(self.local_file(file), self.remote_path(file))

  def get(self, files):
    for file in files:
      if not self.file_exists(file, True):
        print("Get: ", self.remote_path(file), self.local_file(file))
        gfs().get_file(self.remote_path(file), self.local_file(file))

  def upload(self):
    for k, v in self.files.items():
      self.put(v)

  def download(self):
    for k, v in self.files.items():
      self.get(v)

  def vocab_files(self):
    cf, tf = self.files["vocab"]
    return self.local_file(cf), self.local_file(tf)

  def generate_vocabs(self):
    import sys

    from ..params import Max_Vocab
    from .tokenizer import ReservedTokens

    cf, tf = self.vocab_files()
    if not (self.file_exists(cf) and self.file_exists(tf)):
      c_vocab, t_vocab = sp_prep(self.name, self.data)
      extras = ReservedTokens
      write_vocab_file(cf, extras + c_vocab)
      write_vocab_file(tf, extras + t_vocab)

      if len(extras + c_vocab) > Max_Vocab or len(extras + t_vocab) > Max_Vocab:
        logging.error("Vocab size too large")
        sys.exit()

  def create_tokenizers(self):
    cf, tf = self.vocab_files()
    if self.torch:
      from ..torch.simple_tokenizer import SimpleTokenizer

      self.ct = SimpleTokenizer(cf)
      self.tt = SimpleTokenizer(tf)
    else:
      from ..tf.simple_tokenizer import SimpleTokenizer

      self.ct = SimpleTokenizer(cf)
      self.tt = SimpleTokenizer(tf)

  def vocab_sizes(self):
    self.ct.vocab_size(), self.tt.vocab_size()

  def encode_context(self, c):
    return " ".join(self.cm.encode_as_pieces(c))

  def encode_target(self, c):
    return " ".join(self.tm.encode_as_pieces(c))

  def get_transforms(self):
    import torch

    def st(s):
      return torch.tensor(self.ct.tokenize(s, mark=False), dtype=torch.int)

    def tt(s):
      return torch.tensor(self.tt.tokenize(s, mark=False), dtype=torch.int)

    return st, tt, self.tt.vocab


if __name__ == "__main__":
  logging.basicConfig(level=logging.WARNING)
  logging.basicConfig(level=logging.INFO)
  name = "playlists_uri"
  data = "ml-sketchbook.cogs_data.playlists_uri_1m"
  p = Prep(name, data)
  print("P Vocab", p.vocab_sizes())
