import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("data/essays.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text)
