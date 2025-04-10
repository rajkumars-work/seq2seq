from pathlib import Path

import torch
from transformers import (
  AutoModelForCausalLM,
  AutoModelForSeq2SeqLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  DataCollatorForSeq2Seq,
)

# pip install torch transformers datasets accelerate peft bitsandbytes


Model_Dir = "/tmp/hf/models/"
Cache_Dir = "/tmp/hf/cache/"
Finetune_Dir = "/tmp/hf/finetuned/"

Base_Model = "t5/test"
Default_Model = Model_Dir + Base_Model
Gcs_Model_Path = "gs://cogs_models/hf/"


def model_path(name: str):
  return Model_Dir + name + "/"


def model_name(name: str, size: str = "small"):
  pref = {"small": "s", "base": "b", "large": "l", "3B": "h"}
  return f"t5{pref[size]}.{name}"


def translate(text, model, tokenizer):
  inputs = tokenizer(text, return_tensors="pt").input_ids
  outputs = model.generate(
    inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95
  )
  return tokenizer.decode(outputs[0], skip_special_tokens=True)


# - io -----
# download model from gcs
def download_model(name: str):
  from ..utils.gcs import download_prefix

  src = Gcs_Model_Path + name
  target = Model_Dir + name
  Path(target).mkdir(parents=True, exist_ok=True)
  download_prefix(src, target)


def upload_model(name: str):
  from ..utils.gcs import gfs

  target = Gcs_Model_Path + name + "/"
  dir = model_path(name)
  if Path(dir).is_dir():
    gfs().put(dir, target, recursive=True)


def download_and_load(name: str, quantize: bool = False, s2s: bool = False):
  download_model(name)
  return load_model(name, quantize=quantize, s2s=s2s)


# Load tokenizer and model
def save_model(model, tokenizer, m_name):
  save_path = model_path(m_name)
  model.save_pretrained(save_path)
  tokenizer.save_pretrained(save_path)
  ms = model_size_gb(model)
  print(f"Saved to {save_path} with size {ms} GB")


def load_tokenizer(model_name):
  return AutoTokenizer.from_pretrained(
    model_name, cache_dir=Cache_Dir, trust_remote_code=True
  )


def load_model(model_name, quantize: bool = False, s2s: bool = False):
  quantization = bit_config() if quantize else None
  tokenizer = AutoTokenizer.from_pretrained(
    model_name, cache_dir=Cache_Dir, trust_remote_code=True
  )
  loader = AutoModelForSeq2SeqLM if s2s else AutoModelForCausalLM
  model = loader.from_pretrained(
    model_name,
    quantization_config=quantization,
    device_map=None if s2s else "auto",
    cache_dir=Cache_Dir,
    trust_remote_code=True,
  )
  ms = model_size_gb(model)
  print(f"Loaded model {model_name} with size {ms} GB")
  return model, tokenizer


def cache_path(name):
  parts = name.split("/")
  return Cache_Dir + f"models--{parts[0]}--{parts[1]}"


# - quantization ------------------
# Configure 4-bit quantization
def bit_config():
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for faster computation
  )
  return bnb_config


# - data --
def collator(model_name_or_path: str, s2s: bool = True):
  model, tokenizer = load_model(model_name_or_path, s2s)
  collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name_or_path)
  return model, tokenizer, collator


# - utils --
def model_size_gb(model):
  return model.get_memory_footprint() / (1024 * 1024 * 1024)


def model_size_str(model):
  return f"Model size is {model_size_gb(model)} GB"
