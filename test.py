import torch
from PIL import Image
import requests
from model import ViT5, ViT5Config
from transformers import ViTFeatureExtractor, T5TokenizerFast



feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = T5TokenizerFast.from_pretrained("unicamp-dl/ptt5-base-portuguese-vocab")
config = ViT5Config()
config.decoder_start_token_id=0

model = ViT5.from_pretrained("./checkpoint")



url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

prompt = "Two cute cats"
input_ids = tokenizer(
    prompt, return_tensors="pt"
).input_ids

# inputs = {"input_ids": input_ids, "pixel_values": pixel_values}
print(input_ids.shape, pixel_values.shape)
print(model.config)
outputs = model.generate(input_ids, pixel_values=pixel_values, max_length=64, min_length=20)
print(outputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
