import torch
import requests
from PIL import Image
from transformers import ViTFeatureExtractor, AutoTokenizer, \
                         VisionEncoderDecoderModel

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'



encoder_checkpoint = 'google/vit-base-patch16-224'
decoder_checkpoint = 'surajp/gpt2-hindi'
model_checkpoint = 'team-indain-image-caption/hindi-image-captioning'
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)

img_url = '00172.png'
raw_image = Image.open(img_url).convert('RGB')

#Inference
sample = feature_extractor(raw_image, return_tensors="pt").pixel_values.to(device)
clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]

caption_ids = model.generate(sample, max_length = 50)[0]
caption_text = clean_text(tokenizer.decode(caption_ids))
print(caption_text)
