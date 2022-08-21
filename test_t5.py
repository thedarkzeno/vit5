import torch
from PIL import Image
import requests
# from model import ViT5Model, ViT5Config
from transformers import T5TokenizerFast, T5Model, T5Config

tokenizer = T5TokenizerFast.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")
config = T5Config()
model = T5Model.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")
model.save_pretrained("./checkpoint")

prompt = "Two cute cats"
input_ids = tokenizer(
    prompt, return_tensors="pt"
).input_ids


model(input_ids)
