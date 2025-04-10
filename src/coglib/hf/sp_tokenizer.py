# Needed numpy==1.26.4

from pathlib import Path

from tokenizers import SentencePieceBPETokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .cog_ds import cogs_ds
from .models import model_name, model_path

special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
vocab_size = 32000


class SPTokenizer:
  def __init__(self, name: str):
    self.path = model_path(model_name(name))
    self.tpath = self.path + "/tokenizer.json"

    if not Path(self.path).is_dir():
      self.train()
    self.tokenizer = AutoTokenizer.from_pretrained(self.path)

  def special_symbols(self):
    return dict((s, self.tokenizer.token_to_id(s)) for s in special_tokens)

  def train_sp(self):
    # combines context and target. Should we have two
    def get_batch():
      for i in range(0, len(ds), 1000):
        t = ds[i : i + 1000]["target"]
        c = ds[i : i + 1000]["context"]
        yield c + t

    ds = cogs_ds(self.name)["train"]

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train_from_iterator(
      get_batch(),
      vocab_size=vocab_size,
      min_frequency=2,
      show_progress=True,
      special_tokens=special_tokens,
    )
    t = PreTrainedTokenizerFast(
      tokenizer_object=tokenizer,
      model_max_length=vocab_size,
      special_tokens=special_tokens,
    )

    t.bos_token = "<s>"
    t.bos_token_id = tokenizer.token_to_id("<s>")
    t.pad_token = "<pad>"
    t.pad_token_id = tokenizer.token_to_id("<pad>")
    t.eos_token = "</s>"
    t.eos_token_id = tokenizer.token_to_id("</s>")
    t.unk_token = "<unk>"
    t.unk_token_id = tokenizer.token_to_id("<unk>")
    t.cls_token = "<cls>"
    t.cls_token_id = tokenizer.token_to_id("<cls>")
    t.sep_token = "<sep>"
    t.sep_token_id = tokenizer.token_to_id("<sep>")
    t.mask_token = "<mask>"
    t.mask_token_id = tokenizer.token_to_id("<mask>")

    tokenizer.save(self.tpath)
    t.save_pretrained(self.path)
    self.tokenizer = t


# -----
def test(train=False):
  name = "por_en"
  t = SPTokenizer(name)
  a, b = "let's test this tokenizer...", "em um par de frases."
  e = t.tokenizer.encode(a, b)
  print("Encoded: ", e)
  print("Decoded: ", t.tokenizer.decode(e))


if __name__ == "__main__":
  test()
