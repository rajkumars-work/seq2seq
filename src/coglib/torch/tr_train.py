import os

import torch
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ..utils.gcs import gfs
from .s2s_data import SData
from .s2s_models import Seq2SeqTransformer, accuracy, create_mask

# torchrun needs absolute paths (run from top level)
# torchrun --standalone --nproc_per_node=<N> src/cogs/torch/tr_train.py

PAD_IDX = SData.PAD_IDX
Max_Steps = 100


class Trainer:
  def __init__(
    self,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: DataLoader,
    path: str,
  ) -> None:
    self.device = self._get_device()
    model = model.to(self.device)
    self.model = DDP(model)
    self.train_data = train_data
    self.optimizer = optimizer
    self.path = path
    self.save_every = 5
    self.min_loss = 10.0
    self.start_epoch = 1
    self.pad_idx = PAD_IDX
    self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx)
    self._load_checkpoint()

  def _get_device(self):
    # mps currently does not support DDP
    return int(os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else "cpu"

  def _run_batch(self, src, tgt):
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
    loss.backward()
    self.optimizer.step()

    acc = accuracy(output, tgt_out)
    return loss, acc

  # Note: doesn't use sampling (all gpus get whole dataset)
  def _run_epoch(self, epoch):
    steps = self.train_data.len() if hasattr(self.train_data, "len") else Max_Steps
    # self.train_data.sampler.set_epoch(epoch)
    loss, acc = 0.0, 0.0
    for i, (source, targets) in enumerate(self.train_data, 1):
      if i > Max_Steps:
        print(
          f"[{self.device}] Epoch {epoch} | Batchsize: {len(source)} | Steps: {steps}"
        )
        break
      source = source.to(self.device)
      targets = targets.to(self.device)
      lo, ac = self._run_batch(source, targets)
      loss += lo
      acc += ac
    return loss.item() / i, acc.item() / i

  def _load_checkpoint(self):
    path = cpath(self.name)
    if gfs().exists(path):
      with gfs().open(path, "rb") as fh:
        d = torch.load(fh)
        self.model.module.load_state_dict(d["model_state_dict"])
        self.optimizer.load_state_dict(d["optimizer_state_dict"])
        self.min_loss = d["loss"]
        self.start_epoch = d["epoch"]

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

  def train(self, max_epochs):
    for epoch in range(self.start_epoch, max_epochs + 1):
      loss, acc = self._run_epoch(epoch)
      if epoch % self.save_every == 0:
        if self.device == 0 or self.device in ("cpu", "mpu"):
          self._save_checkpoint(epoch, loss)
          print(f"{self.device}: Epoch {epoch} loss/acc {loss:.1f} / {acc:.1f}")


# rank: Unique identifier of each process,  world_size: Total number of processes
def ddp_setup():
  init_process_group(backend="nccl")
  torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def create_model(path, svs, tvs):
  params = {
    "d_model": 128,
    "dff": 512,
    "num_layers": 6,
    "num_heads": 8,
    "dropout_rate": 0.1,
    "input_vocab_size": svs,
    "target_vocab_size": tvs,
  }
  model = Seq2SeqTransformer(**params)
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
  )
  return model, optimizer


# Note: doesn't use sampling
def load_train_objs(name, data):
  sd = SData(name, data)
  svl, tvl = sd.get_vocab_sizes()
  model, optimizer = create_model(cpath(name), svl, tvl)  # load your model
  train_set, _ = sd.get_dls(dist=False)
  return train_set, model, optimizer


def cpath(name):
  # return f"/tmp/s2s/{name}.pt"
  return f"gs://cogs_models/torch/{name}.pt"


def main(total_epochs, name, data):
  ddp_setup()
  train_data, model, optimizer = load_train_objs(name, data)
  trainer = Trainer(model, optimizer, train_data, cpath(name))
  trainer.train(total_epochs)
  destroy_process_group()


if __name__ == "__main__":
  total_epochs = 30
  name = "por_en"
  data = "ml-sketchbook.cogs_data.por_en"
  main(total_epochs, name, data)
