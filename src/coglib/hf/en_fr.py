from datasets import load_dataset
from transformers import (
  AutoModelForSeq2SeqLM,
  AutoTokenizer,
  DataCollatorForSeq2Seq,
  Seq2SeqTrainer,
  Seq2SeqTrainingArguments,
)

source_lang = "en"
target_lang = "fr"
prefix = "translate English to French: "


books = load_dataset("opus_books", "en-fr")
books = books["train"].train_test_split(test_size=0.2)
books = books["train"].train_test_split(test_size=0.2)

# books[0] = {'id': '90560', 'translation': {'en': 'But this', 'fr': 'Mais ce plateau'}}

checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


def preprocess_function(examples):
  inputs = [prefix + example[source_lang] for example in examples["translation"]]
  targets = [example[target_lang] for example in examples["translation"]]
  model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
  return model_inputs


tokenized_books = books.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

training_args = Seq2SeqTrainingArguments(
  output_dir="/tmp/en_fr/",
  eval_strategy="epoch",
  learning_rate=2e-5,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  weight_decay=0.01,
  save_total_limit=1,
  num_train_epochs=2,
  predict_with_generate=True,
  fp16=True,  # change to bf16=True for XPU
)

trainer = Seq2SeqTrainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_books["train"],
  eval_dataset=tokenized_books["test"],
  processing_class=tokenizer,
  data_collator=data_collator,
  # compute_metrics=compute_metrics,
)

trainer.train()
path = "/tmp/en_fr/"
model.save_pretrained(path)
tokenizer.save_pretrained(path)

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSeq2SeqLM.from_pretrained(path)

text = (
  "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
)
inputs = tokenizer(text, return_tensors="pt").input_ids
outputs = model.generate(
  inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95
)
res = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(res)
