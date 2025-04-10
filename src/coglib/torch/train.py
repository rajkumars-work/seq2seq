from time import time

import torch

from .models import ScheduledOptim, Transformer, masked_accuracy, masked_loss
from .s2s_data import SData
from .utils import cuda_mem, device, load, model_size, save

DEVICE = device
print("Using Device ", DEVICE)
torch.manual_seed(0)

Epochs = 30
Max_Batches = 100

Default_model_params = {
  "d_model": 128,
  "dff": 512,
  "num_layers": 6,
  "num_heads": 8,
  "dropout_rate": 0.1,
}

print("\nUsing Device ", device, "\n")


def create_model(name, params):
  m = Transformer(**params).to(device)
  m = m.to(DEVICE)

  optim = torch.optim.Adam(m.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
  optim = ScheduledOptim(optim, d_model=params["d_model"])
  epoch, loss = load(name, m, optim._optimizer)
  model_size(m)
  return m, optim, epoch, loss


def get_model(name, data):
  sd = SData(name, data, batch_first=True)
  cvs, tvs = sd.get_vocab_sizes()
  data_params = {"input_vocab_size": cvs, "target_vocab_size": tvs}
  params = Default_model_params | data_params
  model, _, _, _ = create_model(name, params)
  return model, sd


# ----
def train_epoch(model, optim, data):
  model.train()
  total_loss, total_acc = 0.0, 0.0
  for i, (c, t) in enumerate(data, 1):
    if i > Max_Batches:
      break
    optim.zero_grad()
    labels = t[:, 1:]
    t = t[:, :-1]

    c = c.to(DEVICE)
    t = t.to(DEVICE)
    labels = labels.to(DEVICE)

    pred = model((c, t))
    loss = masked_loss(labels, pred)
    loss.backward()
    optim.step_and_update_lr()
    acc = masked_accuracy(labels, pred)

    total_loss += loss
    total_acc += acc
    torch.cuda.empty_cache()
  return total_loss / i, total_acc / i


def evaluate(model, data):
  model.eval()
  total_loss, total_acc = 0.0, 0.0
  for i, (c, t) in enumerate(data, 1):
    if i > Max_Batches:
      break
    labels = t[:, 1:]
    t = t[:, :-1]

    c = c.to(DEVICE)
    t = t.to(DEVICE)
    labels = labels.to(DEVICE)

    pred = model((c, t))
    loss = masked_loss(labels, pred)
    acc = masked_accuracy(labels, pred)

    total_loss += loss
    total_acc += acc
    torch.cuda.empty_cache()
  return total_loss / i, total_acc / i


def train(name, m, optim, ds, val_ds, epochs, min_loss):
  for e in range(1, epochs + 1):
    st = time()
    loss, acc = train_epoch(m, optim, ds)
    cuda_mem()
    et = time() - st
    val_loss, val_acc = evaluate(m, val_ds)
    print(
      f"Epoch: {e}({et:.0f}s) Loss(t/e): {loss:.3f} / {val_loss:.3f}, ",
      f"Acc(t/e): {acc:.3f} / {val_acc:.3f}, ",
      flush=True,
    )
    if val_loss < min_loss:
      min_loss = val_loss
      save(name, m, optim._optimizer, e, min_loss)


def run(name, data, epochs=Epochs):
  sd = SData(name, data, batch_first=True)
  cvs, tvs = sd.get_vocab_sizes()
  data_params = {"input_vocab_size": cvs, "target_vocab_size": tvs}
  params = Default_model_params | data_params
  train_dl, val_dl = sd.get_dls()
  model, optimizer, epoch, loss = create_model(name, params)
  cuda_mem()
  train(name, model, optimizer, train_dl, val_dl, epochs, loss)
  return model, sd


def eval(model, sd):
  from itertools import islice

  from .models import CGenerator

  model.eval()
  _, val_ds = sd.get_dss()
  val_ds = islice(val_ds, 10)
  gen = CGenerator(model, sd.ct, sd.tt)
  res = [(gen.predict(c, ""), t) for c, t in val_ds]
  res = [(sd.dp.decode_targets(c), sd.dp.decode_targets(t)) for c, t in res]
  for c, t in res:
    print("\n", c)
    print(t)


if __name__ == "__main__":
  name = "por_en"
  data = "ml-sketchbook.cogs_data.por_en"
  m, sd = run(name, data)
  # m, sd = get_model(name, data)
  eval(m, sd)
