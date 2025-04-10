# Needed numpy==1.26.4

from cog_ds import cogs_ds, o_path
from tokenizers import (
  Tokenizer,
  decoders,
  models,
  normalizers,
  pre_tokenizers,
  processors,
  trainers,
)
from transformers import PreTrainedTokenizerFast

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
vocab_size = 32000


def t_path(name):
  return o_path(name) + "/tokenizer.json"


def t_train(name):
  # combines context and target. Should we have two ?
  def get_batch():
    for i in range(0, len(ds), 1000):
      t = ds[i : i + 1000]["target"]
      c = ds[i : i + 1000]["context"]
      yield c + t

  ds = cogs_ds(name)["train"]

  tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
  tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
  tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
  trainer = trainers.WordPieceTrainer(
    vocab_size=vocab_size, special_tokens=special_tokens
  )
  tokenizer.train_from_iterator(get_batch(), trainer=trainer)
  cls_token_id = tokenizer.token_to_id("[CLS]")
  sep_token_id = tokenizer.token_to_id("[SEP]")
  tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS]:0 $A:0 [SEP]:0",
    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
  )
  tokenizer.decoder = decoders.WordPiece(prefix="##")
  tokenizer.save(t_path(name))
  return tokenizer


def t_load(name):
  tpath = t_path(name)
  # t =  Tokenizer.from_file( tpath )
  wt = PreTrainedTokenizerFast(
    # tokenizer_object=t,
    tokenizer_file=tpath,  # You can load from the tokenizer file, alternatively
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
  )
  return wt


# -----


def test(train=False):
  name = "por_en"
  if train:
    t_train(name)
  tokenizer = t_load(name)
  a, b = "Let's test this tokenizer...", "on a pair of sentences."
  e = tokenizer.encode(a, b)
  print(e)
  print("Decoded: ", tokenizer.decode(e))


if __name__ == "__main__":
  test()
