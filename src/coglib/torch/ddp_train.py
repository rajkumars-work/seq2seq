import logging
import os
import sys
from itertools import islice
from time import time

import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..utils.gcs import gfs
from ..utils.text_io import upload_dicts_to_bq
from .s2s_data import SData
from .s2s_models import accuracy, create_mask, translate
from .utils import create_model_and_optimizer, memory

PAD_IDX = SData.PAD_IDX


class Trainer:
  def __init__(
    self,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: DataLoader,
    val_data: DataLoader,
    rank: int,
    name: str,
  ) -> None:
    self.device = self._get_device(rank)
    model = model.to(self.device)
    self.model = DDP(model)
    self.train_data = train_data
    self.val_data = val_data
    self.optimizer = optimizer
    self.name = name
    self.loss = 100.0  # large value
    self.acc = 0
    self.epochs = 0
    self.start_epoch = 0
    self.save_every = 10
    self.pad_idx = PAD_IDX
    self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx)
    self._load_checkpoint()

  def _get_device(self, rank):
    # mps currently does not support DDP
    return rank if torch.cuda.is_available() else "cpu"

  def _run_batch(self, src, tgt, train: bool = True):
    if train:
      self.optimizer.zero_grad()

    tgt_input = tgt[:-1, :]
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
      src, tgt_input, self.device
    )
    output = self.model(
      src,
      tgt_input,
      src_mask,
      tgt_mask,
      src_padding_mask,
      tgt_padding_mask,
      src_padding_mask,
    )
    tgt_out = tgt[1:, :].to(torch.int64)

    output = output.reshape(-1, output.shape[-1])
    tgt_out = tgt_out.reshape(-1)

    loss = self.loss_fn(output, tgt_out)
    if train:
      loss.backward()
      self.optimizer.step()

    acc = accuracy(output, tgt_out)
    return loss, acc

  def _run_epoch(self, epoch, train: bool = True):
    data = self.train_data if train else self.val_data
    self.model.train() if train else self.model.eval()

    steps = len(data)
    batch_size = len(next(iter(data))[0])

    data.sampler.set_epoch(epoch)
    loss, acc = 0.0, 0.0
    for i, (source, targets) in enumerate(data, 1):
      lo, ac = self._run_batch(source.to(self.device), targets.to(self.device), train)
      loss = loss + lo.item()
      acc = acc + ac.item()
      if i % 1000 == 0:
        print(
          f"{i} Batch/steps: {len(source)}/{steps}\t loss/acc: {(loss / i):.2f}",
          f" / {(acc / i):.2f}\t{memory()}",
          flush=True,
        )
      source, target, lo, ac = None, None, None, None
      del target
    loss = loss / i
    acc = acc / i
    print(
      f"[{self.device}] Epoch {epoch} | steps/batches: {steps}/{batch_size}\t",
      f" loss/acc: {loss:.2f} / {acc:.2f}",
    )
    return loss, acc

  def _load_checkpoint(self):
    path = cpath(self.name)
    if gfs().exists(path):
      with gfs().open(path, "rb") as fh:
        d = torch.load(fh)
        self.model.module.load_state_dict(d["model_state_dict"])
        self.optimizer.load_state_dict(d["optimizer_state_dict"])
        self.loss = d["loss"]
        self.start_epoch = d["epoch"] if d["epoch"] else 0

  def _save_checkpoint(self, epoch, loss):
    ckp = {
      "epoch": epoch,
      "loss": loss,
      "model_state_dict": self.model.module.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
    }
    if self.device == 0 or self.device in ("cpu", "mpu"):
      print(f"[{self.device}] Saving to {cpath(self.name)}")
      path = cpath(self.name)
      with gfs().open(path, "wb") as fh:
        torch.save(ckp, fh)

  def _save(self):
    from .utils import save

    if self.device == 0 or self.device in ("cpu", "mpu"):
      save(self.name, self.model.module, self.loss, self.acc, self.epochs)

  def train(self, max_epochs):
    print("Epochs: done/last: ", self.start_epoch, "/", max_epochs)
    loss, acc = 0, 0
    for epoch in range(self.start_epoch + 1, max_epochs + 1):
      st = time()
      loss, acc = self._run_epoch(epoch)
      et = time() - st
      if epoch % self.save_every == 0:
        if self.device == 0 or self.device in ("cpu", "mpu"):
          self._save_checkpoint(epoch, loss)
          print(
            f"[{self.device}] Epoch {epoch} | {et:.0f}s\t loss/acc {loss:.2f} /",
            f" {acc:.2f} ",
          )
    self.loss = loss
    self.acc = acc
    self.epochs = max(self.start_epoch, max_epochs)
    if max_epochs > self.start_epoch:
      if self.device == 0 or self.device in ("cpu", "mpu"):
        self._save()

  def evaluate(self):
    if self.device == 0 or self.device in ("cpu", "mpu"):
      loss, acc = self._run_epoch(0, False)
      print(f"[{self.device}] Eval loss/acc {loss:.2f} / {acc:.2f}")
      return loss, acc
    else:
      return 0, 0

  def eval(self, sd):
    if self.device == 0 or self.device in ("cpu", "mpu"):
      _, data = sd.get_dss()
      data = islice(data, 10)
      st, tt, _ = sd.get_transforms()
      tokens = [
        (translate(self.model.module, c, st, tt, self.device), c, t) for c, t in data
      ]
      res = [
        (
          sd.pr.tm.decode_pieces(r.split()),
          sd.pr.cm.decode_pieces(c.split()),
          sd.pr.tm.decode_pieces(t.split()),
        )
        for r, c, t in tokens
      ]

      # uri -> isrc intermediate processing
      uri_prep = sd.pr.is_uri_prep()
      res = [
        (
          sd.pr.isrc.from_isrc_str(r) if uri_prep else r,
          sd.pr.isrc.from_isrc_str(c) if uri_prep else c,
          sd.pr.isrc.from_isrc_str(t) if uri_prep else t,
        )
        for r, c, t in res
      ]
      res = [{"pred": r, "context": c, "target": t} for r, c, t in res]
      return res
    else:
      return []


# rank: Unique identifier of each process,  world_size: Total number of processes
def ddp_setup(rank: int, world_size: int):
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  torch.cuda.device(rank)
  init_process_group(backend="gloo", rank=rank, world_size=world_size)


def load_train_objs(name, data, uri_prep=True):
  logging.info(f"Checking data for {name}, {data}")
  sd = SData(name, data, uri_prep=uri_prep)
  cvs, tvs = sd.get_vocab_sizes()
  model, optimizer = create_model_and_optimizer(cvs, tvs)  # load your model
  train_set, val_set = sd.get_dls(dist=True)
  return train_set, val_set, model, optimizer, sd


def cpath(name):
  return f"gs://cogs_models/torch/{name}.pt"


def mpath(name):
  return f"gs://cogs_models/torch/{name}_full.pt"


def main(rank, world_size, name, data, total_epochs, uri_prep):
  logging.basicConfig(level=logging.INFO)
  logging.info(f"Starting {rank}")
  ddp_setup(rank, world_size)
  train_data, val_data, model, optimizer, sd = load_train_objs(name, data, uri_prep)
  trainer = Trainer(model, optimizer, train_data, val_data, rank, name)
  trainer.train(total_epochs)
  if rank == 0:
    ds = trainer.eval(sd)
    upload_dicts_to_bq(ds, name)
  destroy_process_group()


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  args = sys.argv
  name = args[1] if len(args) > 1 else "por_en_r"
  data = args[2] if len(args) > 2 else "ml-sketchbook.cogs_data.por_en"
  epochs = int(float(args[3])) if len(args) > 3 else 300
  uri_prep = False if len(args) > 4 and args[4].lower().startswith("f") else True

  world_size = max(torch.cuda.device_count(), 1)
  logging.info(f"Spawning {world_size} processes and {epochs} epochs")
  mp.spawn(main, args=(world_size, name, data, epochs, uri_prep), nprocs=world_size)
