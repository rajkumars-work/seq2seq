from itertools import islice
from timeit import default_timer as timer

import torch
from torch import nn

from .s2s_data import SData
from .s2s_models import Seq2SeqTransformer, accuracy, create_mask, translate
from .utils import device, model_size, save

DEVICE = device
print("Using Device ", DEVICE)
torch.manual_seed(0)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=SData.PAD_IDX)


Epochs = 30
Max_Batches = 300
Pre = "_s2s"

Default_model_params = {
  "d_model": 128,
  "dff": 512,
  "num_layers": 6,
  "num_heads": 8,
  "dropout_rate": 0.1,
}


def create_model(name, params):
  epoch = 0
  loss = 0
  model = Seq2SeqTransformer(**params)
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  model = model.to(DEVICE)
  optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
  )
  # epoch, loss = load_params(name, model, optimizer, DEVICE)
  model_size(model)
  return model, optimizer, epoch, loss


def get_model(name, data):
  sd = SData(name, data, batch_first=True)
  cvs, tvs = sd.get_vocab_sizes()
  data_params = {"input_vocab_size": cvs, "target_vocab_size": tvs}
  params = Default_model_params | data_params
  model, _, _, _ = create_model(name, params)
  return model, sd


def train_epoch(model, optimizer, data):
  model.train()
  losses, accs = 0, 0

  for i, (src, tgt) in enumerate(data, 1):
    if i > Max_Batches:
      break
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:-1, :]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
      src, tgt_input, DEVICE
    )

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


def evaluate(model, data):
  model.eval()
  losses, accs = 0, 0

  for i, (src, tgt) in enumerate(data, 1):
    if i > Max_Batches:
      break
    src = src.to(DEVICE)
    tgt = tgt.to(DEVICE)

    tgt_input = tgt[:-1, :]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
      src, tgt_input, DEVICE
    )

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
def train(name, model, optimizer, train_data, val_data, epochs, loss):
  for epoch in range(1, epochs + 1):
    start_time = timer()
    train_loss, acc = train_epoch(model, optimizer, train_data)
    et = timer() - start_time
    val_loss, val_acc = evaluate(model, val_data)
    print(
      f"Epoch: {epoch}({et:.0f}s) Loss(t/e): {train_loss:.3f} / {val_loss:.3f}, ",
      f"Acc(t/e): {acc:.3f} / {val_acc:.3f}, ",
      flush=True,
    )
    if val_loss < loss:
      loss = val_loss
      save(name, model, optimizer, epoch, loss, Pre)


def eval(model, sd):
  _, data = sd.get_dss()
  data = islice(data, 10)
  st, tt, _ = sd.get_transforms()
  res = [(translate(model, c, st, tt, DEVICE), t) for c, t in data]
  res = [(sd.dp.decode_targets(c), sd.dp.decode_targets(t)) for c, t in res]
  for c, t in res:
    print("\n", c)
    print(t)


def run(name, data):
  sd = SData(name, data)
  cvs, tvs = sd.get_vocab_sizes()
  data_params = {"input_vocab_size": cvs, "target_vocab_size": tvs}
  params = Default_model_params | data_params
  train_dl, val_dl = sd.get_dls()
  model, optimizer, epoch, loss = create_model(name, params)
  train(name, model, optimizer, train_dl, val_dl, Epochs, loss)  # train
  eval(model, sd)


if __name__ == "__main__":
  name = "por_en_new"
  data = "ml-sketchbook.cogs_data.por_en"
  m, sd = run(name, data)
  # m, sd = get_model(name, data)
  # eval(m, sd)
