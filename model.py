import os, sys, shutil
import numpy as np
from PIL import Image

import jax
from transformers import ViTFeatureExtractor
from transformers import GPT2Tokenizer
from huggingface_hub import hf_hub_download

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

# Main model -  ViTGPT2LM
from vit_gpt2.modeling_flax_vit_gpt2_lm import FlaxViTGPT2LMForConditionalGeneration

# create target model directory
model_dir = './models/'
os.makedirs(model_dir, exist_ok=True)
# copy config file
filepath = hf_hub_download("flax-community/vit-gpt2", "checkpoints/ckpt_5/config.json")
shutil.copyfile(filepath, os.path.join(model_dir, 'config.json'))
# copy model file
filepath = hf_hub_download("flax-community/vit-gpt2", "checkpoints/ckpt_5/flax_model.msgpack")
shutil.copyfile(filepath, os.path.join(model_dir, 'flax_model.msgpack'))

flax_vit_gpt2_lm = FlaxViTGPT2LMForConditionalGeneration.from_pretrained(model_dir)

vit_model_name = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)

gpt2_model_name = 'asi/gpt-fr-cased-small'
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

max_length = 32
num_beams = 8
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


@jax.jit
def predict_fn(pixel_values):

    return flax_vit_gpt2_lm.generate(pixel_values, **gen_kwargs)

def predict(image):

    # batch dim is added automatically
    encoder_inputs = feature_extractor(images=image, return_tensors="jax")
    pixel_values = encoder_inputs.pixel_values

    # generation
    generation = predict_fn(pixel_values)

    token_ids = np.array(generation.sequences)[0]
    caption = tokenizer.decode(token_ids)
    caption = caption.replace('<s>', '').replace('</s>', '').replace('<pad>', '')
    caption = caption.replace("à l'arrière-plan", '').replace("Une photo noire et blanche d'", '').replace("en arrière-plan", '')
    while '  ' in caption:
        caption = caption.replace('  ', ' ')
    caption = caption.strip()

    return caption

def compile():

    image_path = 'samples/val_000000039769.jpg'
    image = Image.open(image_path)

    caption = predict(image)
    image.close()

def predict_dummy(image):
    
    return 'dummy caption!'

compile()

sample_dir = './samples/'
sample_fns = tuple([f"{int(f.replace('COCO_val2014_', '').replace('.jpg', ''))}.jpg" for f in os.listdir(sample_dir) if f.startswith('COCO_val2014_')])
