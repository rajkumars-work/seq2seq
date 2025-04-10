import evaluate
import numpy as np
from cog_ds import cogs_ds
from transformers import (
  AutoModelForSeq2SeqLM,
  AutoTokenizer,
  DataCollatorForSeq2Seq,
  EarlyStoppingCallback,
  Seq2SeqTrainer,
  Seq2SeqTrainingArguments,
)

source_lang = "pt"
target_lang = "en"
name = "por_en"
prefix = "translate context to target: "

size = "base"
size = "small"
path = f"/tmp/{size}.cogs.por_en/"
checkpoint = f"google-t5/t5-{size}"
tpath = checkpoint
# tpath = model_path(model_name(name))

tokenizer = AutoTokenizer.from_pretrained(tpath)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
metric = evaluate.load("sacrebleu")


def preprocess_function(examples):
  inputs = [prefix + example for example in examples["context"]]
  targets = [example for example in examples["target"]]
  model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
  return model_inputs


def postprocess_text(preds, labels):
  preds = [pred.strip() for pred in preds]
  labels = [[label.strip()] for label in labels]
  return preds, labels


def compute_metrics(eval_preds):
  preds, labels = eval_preds
  if isinstance(preds, tuple):
    preds = preds[0]
  decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

  decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
  result = metric.compute(predictions=decoded_preds, references=decoded_labels)
  return {"bleu": round(result["score"], 2)}


def run(epochs: int = 1):
  ds = cogs_ds("por_en")
  # ds[0] = {'context': 'But this', 'target': 'Mais ce plateau'}
  ds = ds["train"].train_test_split(test_size=0.2)

  tokenized_ds = ds.map(preprocess_function, batched=True)
  data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

  training_args = Seq2SeqTrainingArguments(
    output_dir=path,
    eval_strategy="steps",
    eval_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=epochs,
    predict_with_generate=True,
    fp16=True,  # change to bf16=True for XPU
    load_best_model_at_end=True,
    metric_for_best_model="loss",
  )

  trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
  )

  trainer.train()
  model.save_pretrained(path)
  tokenizer.save_pretrained(path)


def test():
  tokenizer = AutoTokenizer.from_pretrained(path)
  model = AutoModelForSeq2SeqLM.from_pretrained(path)

  text = (
    prefix
    + "As leguminosas compartilham recursos com bactérias fixadoras de nitrogênio."
  )
  inputs = tokenizer(text, return_tensors="pt").input_ids
  for i in range(10):
    outputs = model.generate(
      inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95
    )
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(res)


if __name__ == "__main__":
  run()
  test()
