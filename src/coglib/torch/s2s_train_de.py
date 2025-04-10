import math
from pathlib import Path
from timeit import default_timer as timer

import torch
from torch import nn

from .s2s_models import Seq2SeqTransformer, create_mask, translate
from .utils import device

DEVICE = device
print("Using Device ", DEVICE)
torch.manual_seed(0)

PAD_IDX = 1
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


Epochs = 100
Max_Batches = 500

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


def cpath(name: str = ""):
  C_Path = "/tmp/s2s"
  return f"{C_Path}/{name}model.pt"


# - checkpoints --
def save(path, model, optimizer, epoch, loss):
  torch.save(
    {
      "epoch": epoch,
      "loss": loss,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
    },
    path,
  )


def load(path, model, optimizer) -> int:
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint["model_state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
  epoch = checkpoint["epoch"]
  loss = checkpoint["loss"]
  print("\nLoaded from", path, "Epochs: ", epoch, "Loss: ", loss, "\n")
  return epoch, loss


def create_model(src_vocab_size, tgt_vocab_size, path=None):
  model = Seq2SeqTransformer(
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    EMB_SIZE,
    NHEAD,
    src_vocab_size,
    tgt_vocab_size,
    FFN_HID_DIM,
  )
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  model = model.to(DEVICE)
  optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
  )
  epoch, loss = 1, 10.0
  if Path(cpath()).exists():
    epoch, loss = load(cpath(), model, optimizer)

  return model, optimizer, epoch, loss


def train_epoch(model, optimizer, data):
  model.train()
  losses, accs = 0, 0

  for i, (src, tgt) in enumerate(data, 1):
    if i > Max_Batches:
      break
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:-1, :]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

    logits = model(
      src,
      tgt_input,
      src_mask,
      tgt_mask,
      src_padding_mask,
      tgt_padding_mask,
      src_padding_mask,
    )

    optimizer.zero_grad()

    tgt_out = tgt[1:, :].to(torch.int64)
    logits = logits.reshape(-1, logits.shape[-1])
    tgt_out = tgt_out.reshape(-1)

    # flatten results, then compute loss
    loss = loss_fn(logits, tgt_out)
    loss.backward()
    acc = accuracy(logits, tgt_out)

    optimizer.step()
    losses += loss.item()
    accs += acc.item()

  return losses / i, accs / i


def accuracy(logits, target) -> float:
  la = torch.argmax(logits, -1)
  e = torch.sum((la == target) & (target != PAD_IDX))
  z = torch.sum(target != PAD_IDX)
  res = e / z
  return torch.tensor(0.0) if math.isinf(res) else res


def evaluate(model, data):
  model.eval()
  losses, accs = 0, 0

  for i, (src, tgt) in enumerate(data, 1):
    if i > Max_Batches:
      break
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:-1, :]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

    logits = model(
      src,
      tgt_input,
      src_mask,
      tgt_mask,
      src_padding_mask,
      tgt_padding_mask,
      src_padding_mask,
    )

    tgt_out = tgt[1:, :].to(torch.int64)
    logits = logits.reshape(-1, logits.shape[-1])
    tgt_out = tgt_out.reshape(-1)

    loss = loss_fn(logits, tgt_out)
    acc = accuracy(logits, tgt_out)
    losses += loss.item()
    accs += acc.item()

  return losses / i, accs / i


# ---
def train(model, optimizer, train_data, val_data, epochs, loss):
  for epoch in range(1, epochs + 1):
    start_time = timer()
    train_loss, acc = train_epoch(model, optimizer, train_data)
    end_time = timer()
    val_loss, val_acc = evaluate(model, val_data)
    print(
      f"Epoch: {epoch}, Loss(t/e): {train_loss:.3f} / {val_loss:.3f}, ",
      f"Acc(t/e): {acc:.3f} / {val_acc:.3f}, ",
      f"Epoch time = {(end_time - start_time):.3f}s",
    )
    if val_loss < loss:
      loss = val_loss
      save(cpath(), model, optimizer, epoch, loss)


def eval(model, data, st, tt, post_process):
  res = [translate(model, d, st, tt) for d in data]
  res = [post_process(r) for r in res]
  return res


def run(por: bool = True):
  if por:
    from .s2s_data_por import get_data, get_transforms, post_process

    data = test_data()
  else:
    from .s2s_data_de import get_data, get_transforms, post_process

    data = ["Eine Gruppe von Menschen steht vor einem Iglu ."]

  train_dl, val_dl, src_vocab_size, tgt_vocab_size = get_data()  # data
  model, optimizer, epoch, loss = create_model(src_vocab_size, tgt_vocab_size)  # model
  train(model, optimizer, train_dl, val_dl, Epochs, loss)  # train
  st, tt, _ = get_transforms()
  res = eval(model, data, st, tt, post_process)  # eval
  for r in res:
    print(r)


def test_data():
  from .datasets import CogsDataset

  name = "por_en"
  data = "ml-sketchbook.cogs_data.por_en"
  val_ds = CogsDataset(name, data, 0, 10)
  return [c for c, t in val_ds]


if __name__ == "__main__":
  run()
