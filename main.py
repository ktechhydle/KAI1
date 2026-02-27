import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

with open("../../data/essays.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = tokenizer.encode(text)
