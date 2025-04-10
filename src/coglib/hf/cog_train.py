import evaluate
import numpy as np
from transformers import (
  AutoModelForSeq2SeqLM,
  AutoTokenizer,
  DataCollatorForSeq2Seq,
  EarlyStoppingCallback,
  Seq2SeqTrainer,
  Seq2SeqTrainingArguments,
)

from ..utils.prep import Prep
from ..utils.text_io import upload_dicts_to_bq
from .cog_ds import cogs_ds
from .models import model_name, model_path, translate, upload_model


class Trainer:
  def __init__(self, model, tokenizer, collator, train_ds, val_ds, name):
    self.model = model
    self.tokenizer = tokenizer
    self.collator = collator
    self.train_ds = train_ds
    self.val_ds = val_ds
    self.prefix = "translate context to target: "
    self.metric = evaluate.load("sacrebleu")
    self.name = name
    self.output_path = model_path(model_name(name))

  def preprocess_function(self, examples):
    inputs = [self.prefix + example for example in examples["context"]]
    targets = [example for example in examples["target"]]
    model_inputs = self.tokenizer(
      inputs, text_target=targets, max_length=128, truncation=True
    )
    return model_inputs

  def postprocess_text(self, preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

  def compute_metrics(self, eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
      preds = preds[0]
    decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
    result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": round(result["score"], 2)}

  def train(self, epochs: int = 1):
    tokenized_train_ds = self.train_ds.map(self.preprocess_function, batched=True)
    tokenized_val_ds = self.val_ds.map(self.preprocess_function, batched=True)

    training_args = Seq2SeqTrainingArguments(
      output_dir=self.output_path,
      eval_strategy="steps",
      eval_steps=50,
      learning_rate=2e-5,
      per_device_train_batch_size=32,
      per_device_eval_batch_size=32,
      weight_decay=0.01,
      save_total_limit=1,
      num_train_epochs=epochs,
      predict_with_generate=True,
      fp16=True,  # change to bf16=True for XPU
      load_best_model_at_end=True,
      metric_for_best_model="loss",
    )

    trainer = Seq2SeqTrainer(
      model=self.model,
      args=training_args,
      train_dataset=tokenized_train_ds,
      eval_dataset=tokenized_val_ds,
      processing_class=self.tokenizer,
      data_collator=self.collator,
      compute_metrics=self.compute_metrics,
      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    self.model.save_pretrained(self.output_path)
    upload_model(model_name(self.name))

  def translate(self, text):
    return translate(self.prefix + text, self.model, self.tokenizer)

  def evaluate(self):
    self.tokenizer = AutoTokenizer.from_pretrained(self.output_path)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.output_path)

    ds = self.val_ds[:10]
    cs = ds["context"]
    ts = ds["target"]
    return [
      {"pred": self.translate(c), "context": c, "target": t} for c, t in zip(cs, ts)
    ]


def get_model(name, size):
  checkpoint = f"google-t5/t5-{size}"
  model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
  # output_path = model_path(model_name(name, size))
  # tokenizer = AutoTokenizer.from_pretrained(output_path)
  tokenizer = AutoTokenizer.from_pretrained(checkpoint)
  collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
  return model, tokenizer, collator


def load_train_objs(name, data):
  _ = Prep(name, data)
  ds = cogs_ds(name)
  ds = ds["train"].train_test_split(test_size=0.2)
  model, tokenizer, collator = get_model(name, size="small")
  return model, tokenizer, collator, ds["train"], ds["test"]


def train():
  import sys

  args = sys.argv

  name = args[1] if len(args) > 1 else "playlist_2m"
  data = args[2] if len(args) > 2 else "ml-sketchbook.cogs_data.playlist_isrc_2m"
  epochs = int(float(args[3])) if len(args) > 3 else 1

  model, tokenizer, collator, train_ds, val_ds = load_train_objs(name, data)
  trainer = Trainer(model, tokenizer, collator, train_ds, val_ds, name)
  trainer.train(epochs)
  trainer.save()
  upload_dicts_to_bq(trainer.evaluate(), name)


if __name__ == "__main__":
  import logging

  logging.basicConfig(level=logging.WARNING)
  train()  # needs to be present
