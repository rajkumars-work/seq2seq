import logging

from datasets import load_dataset

from .cog_ds import cogs_ds
from .models import (
  download_and_load,
  load_tokenizer,
  model_path,
  save_model,
  translate,
  upload_model,
)
from .t5 import checkpoint, train

langs = {
  "en": "English",
  "eo": "Esperanto",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "hu": "Hungarian",
  "it": "Italian",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
}


class Translate:
  def __init__(self, source_lang: str = "pt", target_lang: str = "en"):
    if source_lang not in langs or target_lang not in langs:
      return
    self.source_lang = source_lang
    self.target_lang = target_lang
    m_name = model_name(source_lang, target_lang)
    m_path = model_path(m_name)
    self.model, self.tokenizer = download_and_load(m_path, s2s=True)
    self.prefix = prefix(source_lang, target_lang)

  def translate(self, text):
    return translate(self.prefix + text, self.model, self.tokenizer)


def model_name(source_lang, target_lang):
  return f"t5s/{source_lang}-{target_lang}"


def train_lang(source_lang, target_lang, tds):
  m_name = model_name(source_lang, target_lang)
  output_dir = model_path(m_name)
  model, tokenizer = train(output_dir, tds["train"], tds["test"])
  save_model(model, tokenizer, m_name)
  upload_model(m_name)


# data --
def prefix(source_lang, target_lang):
  return f"translate {langs[source_lang]} to {langs[target_lang]}: "


def lang_data(source_lang, target_lang, tokenizer=None, file=None):
  pfix = prefix(source_lang, target_lang)
  tokenizer = tokenizer if tokenizer else load_tokenizer(checkpoint)

  def preprocess_function(examples):
    inputs = [pfix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(
      inputs, text_target=targets, max_length=128, truncation=True
    )
    return model_inputs

  books = (
    load_dataset(file)
    if file
    else load_dataset("opus_books", f"{source_lang}-{target_lang}")
  )
  books = books["train"].train_test_split(test_size=0.2)
  tokenized_books = books.map(preprocess_function, batched=True)
  return tokenized_books


def evaluate(name, t, eval_size=10):
  ds = cogs_ds(name)["train"][:eval_size]
  cs = ds["context"]
  ts = ds["target"]
  return [{"pred": t.translate(c), "context": c, "target": t} for c, t in zip(cs, ts)]


def train_langs(s_lang, t_lang):
  f = f"/home/rkumar/.cache/huggingface/datasets/opus_books/{s_lang}-{t_lang}"
  tds = lang_data(s_lang, t_lang, file=f)
  train_lang(s_lang, t_lang, tds)
  test_langs(s_lang, t_lang)


def test_langs(s_lang, t_lang):
  m = Translate(s_lang, t_lang)
  en = "Legumes share resources with nitrogen-fixing bacteria."
  print(m.translate(en))
  # pt = "As leguminosas compartilham recursos com bactérias fixadoras de nitrogênio."
  # for e in evaluate("por_en", m):
  #  print(e)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  s_lang, t_lang = "en", "pt"
  train_langs(s_lang, t_lang)
