import io
import logging
import sys
from typing import Iterator, List

from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

from ..utils.gcs import gfs
from .text_io import get_json, prefix

# create models and store to gcs


def paths(name):
  cpath = prefix(name) + "/context.model"
  tpath = prefix(name) + "/target.model"
  return cpath, tpath


def create_models(name: str, data: str):
  cpath, tpath = paths(name)

  vocab_len = max_vocab_len(name, data)
  logging.info(f"Setting max vocab length for {name} to {vocab_len}")

  if not gfs().exists(cpath):
    cit = (e["context"] for e in get_json(name, data))
    cm = create_sp_model(cit, vocab_len, cpath)
    logging.info(f"Created context model vocab: {cm.vocab_size()}")

  if not gfs().exists(tpath):
    tit = (e["target"] for e in get_json(name, data))
    tm = create_sp_model(tit, vocab_len, tpath)
    logging.info(f"Created target model vocab: {tm.vocab_size()}")


def getVocab(sp: SentencePieceProcessor) -> List[str]:
  return [sp.IdToPiece(i) for i in range(sp.vocab_size())]


def sp_tokenizers(name: str):
  cpath, tpath = paths(name)
  csp = SentencePieceProcessor(model_proto=gfs().read_bytes(cpath))
  tsp = SentencePieceProcessor(model_proto=gfs().read_bytes(tpath))

  return csp, tsp


def create_sp_model(
  it: Iterator[str], vocab_len: int, upload_path: str
) -> SentencePieceProcessor:
  logging.info(f"SP tokenizer training start with vocab size {vocab_len}")
  model = io.BytesIO()
  SentencePieceTrainer.train(
    sentence_iterator=it,
    model_writer=model,
    vocab_size=vocab_len,
    input_sentence_size=1000000,
    shuffle_input_sentence=True,
  )
  logging.info(f"Uploading to {upload_path}")
  gfs().write_bytes(upload_path, model.getvalue())
  logging.info("Uploading done")
  return SentencePieceProcessor(model_proto=model.getvalue())


# - --------------------------
def tokenize(sp: SentencePieceProcessor, s: str) -> List[str]:
  return sp.encode(s, out_type=str)


def tokenizes(sp: SentencePieceProcessor, s: str) -> str:
  return " ".join(tokenize(sp, s))


def detokenize(sp: SentencePieceProcessor, subwords: List[str]) -> str:
  return sp.decode(subwords)


def detokenizes(sp: SentencePieceProcessor, subwords: str) -> str:
  return sp.decode(subwords.split(" "))


def run():
  args = sys.argv

  name = args[1] if len(args) > 1 else "por_en_vai"
  data = args[2] if len(args) > 2 else "ml-sketchbook.cogs_data.por_en"

  logging.info(f"Sputils: {name}:{data}")

  create_models(name, data)
  csp, tsp = sp_tokenizers(name, data)

  print(tokenizes(tsp, "This is a test string for tokenization"))


Max_Vocab_Length = 32000
Max_Vocab_Length = 15000


def max_vocab_len(name: str, data: str):
  import itertools

  d = itertools.islice(get_json(name, data), 1000000)
  data_len = min(sum(1 for e in d), 1000000)
  max_vs = int(data_len / 4)
  if max_vs < 1024:
    logging.error(f"Vocab Length for {name}, {data} is {max_vs}")
  return min(Max_Vocab_Length, max_vs)


def max_tokens(name: str, data: str):
  csp, tsp = sp_tokenizers(name, data)
  mt = 0
  for e in get_json(name, data):
    c = len(csp.encode(e["context"]))
    t = len(tsp.encode(e["target"]))
    mt = c if c > mt else mt
    mt = t if t > mt else mt

  return mt


def get_tokens(name: str, data: str):
  import json

  csp, tsp = sp_tokenizers(name, data)
  fh = open(f"/tmp/{name}.jsons", "w")
  for e in get_json(name, data):
    c = csp.encode(e["context"])
    c = " ".join(str(i) for i in c)
    t = tsp.encode(e["target"])
    t = " ".join(str(i) for i in t)
    d = {"context": c, "target": t}
    s = json.dumps(d) + "\n"
    fh.write(s)
  print(f"Done: vocabs: context {csp.vocab_size()}, target {tsp.vocab_size()}")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  name = "foobar"
  data = "ml-sketchbook.cogs_data.por_en"
  create_models(name, None)
  # get_tokens(name, data)
  # print(max_tokens(name, data))
