from datasets import load_dataset
from transformers import (
  AutoModelForSeq2SeqLM,
  AutoTokenizer,
  DataCollatorForSeq2Seq,
  Seq2SeqTrainer,
  Seq2SeqTrainingArguments,
)

source_lang = "pt"
target_lang = "en"
prefix = "translate Portuguese to English: "

checkpoint = "google-t5/t5-small"
path = "/tmp/pt_en/"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


def preprocess_function(examples):
  inputs = [prefix + example[source_lang] for example in examples["translation"]]
  targets = [example[target_lang] for example in examples["translation"]]
  model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
  return model_inputs


def run():
  # books[0] = {'id': '90560', 'translation': {'en': 'But this', 'fr': 'Mais ce plateau'}}
  books = load_dataset("opus_books", "en-pt")
  books = books["train"].train_test_split(test_size=0.2)

  tokenized_books = books.map(preprocess_function, batched=True)
  data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

  training_args = Seq2SeqTrainingArguments(
    output_dir="/tmp/pt_en/",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=30,
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
  model.save_pretrained(path)
  tokenizer.save_pretrained(path)


def test():
  tokenizer = AutoTokenizer.from_pretrained(path)
  model = AutoModelForSeq2SeqLM.from_pretrained(path)

  text = "translate Potuguese to English: As leguminosas compartilham recursos com bactérias fixadoras de nitrogênio."
  inputs = tokenizer(text, return_tensors="pt").input_ids
  for i in range(10):
    outputs = model.generate(
      inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95
    )
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(res)


if __name__ == "__main__":
  test()
