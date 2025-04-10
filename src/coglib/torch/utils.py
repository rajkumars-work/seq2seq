import psutil
import torch
from torch import nn

from ..utils.gcs import get_files, gfs, rm_file
from .s2s_models import Seq2SeqTransformer, predict, translate


# - devices --
def get_device():
  device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
  )
  return torch.device(device)


device = get_device()


# - io ---
Prefix = "gs://cogs_models/torch/"


def mpath(name: str):
  return f"{Prefix}{name}_full.pt"


def cpath(name: str):
  return f"{Prefix}{name}.pt"


def list_models():
  return [p[:-8] for p in get_files(Prefix, "_full.pt")]


def rm_model(name):
  rm_file(f"Prefix{name}_full.pt")


def create_model_and_optimizer(cvs, tvs):
  from ..params import Default_Model_Params, Max_Vocab

  params = Default_Model_Params
  params["input_vocab_size"] = cvs if cvs else Max_Vocab
  params["target_vocab_size"] = tvs if tvs else Max_Vocab

  model = Seq2SeqTransformer(**params)
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
  )
  return model, optimizer


# - checkpoints --
def save_params(name, model, optimizer, loss, epochs):
  ckp = {
    "epoch": epochs,
    "loss": loss,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
  }
  path = cpath(name)
  with gfs().open(path, "wb") as fh:
    torch.save(ckp, fh)
  print(f"Saved to {path}: epoch: {epochs},  loss: {loss:0.2f}")


# returns loss, epochs
def load_params(name, model, optimizer, device=device):
  path = cpath(name)
  if gfs().exists(path):
    with gfs().open(path, "rb") as fh:
      d = torch.load(fh, map_location=device)
      model.load_state_dict(d["model_state_dict"])
      optimizer.load_state_dict(d["optimizer_state_dict"])
      epoch = d["epoch"]
      loss = d["loss"]
      print(f"\nLoaded from {path}:  epochs: {epoch}, Loss: {loss:0.2f}\n")
      return epoch, loss
  else:
    print(path, "does not exist")
    return 0.0, 0


def save(name, model, loss, acc, epochs):
  path = mpath(name)
  ivs = model.input_vocab_size
  tvs = model.target_vocab_size
  d = {
    "epoch": epochs,
    "loss": loss,
    "acc": acc,
    "model_state_dict": model.state_dict(),
    "input_vocab_size": ivs,
    "target_vocab_size": tvs,
  }
  with gfs().open(path, "wb") as fh:
    torch.save(d, fh)
  print(f"Saved model state dict to {path}")


def load(name, device=device):
  path = mpath(name)
  if gfs().exists(path):
    print("Loading model state dict", path)
    with gfs().open(path, "rb") as fh:
      d = torch.load(fh, map_location=device)
      ivs = d["input_vocab_size"]
      tvs = d["target_vocab_size"]
      model, _ = create_model_and_optimizer(ivs, tvs)
      model.load_state_dict(d["model_state_dict"])
      return {
        "model": model.to(device),
        "epoch": d["epoch"],
        "loss": d["loss"],
        "acc": d["acc"],
      }
  else:
    return {}


def load_model(name, device=device):
  d = load(name, device)
  return d.get("model", None)


def model_info(name):
  m = load(name)
  if "model" in m:
    m.pop("model")
  return m


# - preds -------------------
def trans(context: str, model, p) -> str:
  s = p.encode_context(context)
  st, tt, tvt = p.get_transforms()
  tokens = translate(model, s, st, tvt, device)
  out = p.tm.decode_pieces(tokens.split())
  return p.isrc.from_isrc_str(out) if p.uri_prep else out


def pred(context: str, tgt: str, model, p) -> str:
  s = p.encode_context(context)
  t = p.encode_target(tgt)
  st, tt, tvt = p.get_transforms()
  tokens = predict(model, s, t, st, tt, tvt, device)
  out = p.tm.decode_pieces(tokens.split())
  return p.isrc.from_isrc_str(out) if p.uri_prep else out


# - memory -------------------
def cuda_mem():
  if torch.cuda.is_available():
    f, a = torch.cuda.mem_get_info()
    a = a // 1000000
    f = f // 1000000
    p = f * 100 // a
    print("Cuda pct, free/avail: ", p, f, a, flush=True)


def model_size(model):
  param_size = 0
  for param in model.parameters():
    param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print("model size: {:.0f} MB".format(size_all_mb))


# def create_default_s2s_model(svs=12851, tvs=12981):
def create_default_s2s_model(device=device):
  from ..params import Default_Model_Params

  params = Default_Model_Params
  model = Seq2SeqTransformer(**params)
  model = model.to(device)
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
  )
  return model, optimizer


def memory():
  d = {"mem_pct": psutil.virtual_memory().percent}
  if torch.cuda.is_available():
    d["cuda_mem"] = torch.cuda.utilization()
  return d


if __name__ == "__main__":
  from ..utils.prep import Prep

  name = "por_en_torch"
  d = load(name)
  m = d["model"]
  p = Prep(name)
  s = "Ola mundo, como esta"
  res = pred(s, "", m, p)
  print("Sentence ", s, "\nPred", res)
