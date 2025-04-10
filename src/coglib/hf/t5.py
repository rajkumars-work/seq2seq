import logging

from models import load_model
from transformers import (
  DataCollatorForSeq2Seq,
  Seq2SeqTrainer,
  Seq2SeqTrainingArguments,
)

sizes = ["small", "base", "large", "3B"]
size = "small"
checkpoint = f"google-t5/t5-{size}"


def load_sizes():
  for size in sizes:
    m, t = load_model(f"google-t5/t5-{size}", s2s=True)


def train(output_dir, tds, eds):
  model, tokenizer = load_model(checkpoint, s2s=True)
  data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

  logging.info("Model Dir " + output_dir)
  training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,  # change to bf16=True for XPU
  )
  trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tds,
    eval_dataset=eds,
    processing_class=tokenizer,
    data_collator=data_collator,
  )
  trainer.train()
  return model, tokenizer


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  load_sizes()
