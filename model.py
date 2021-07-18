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

def predict(image):
    return 'dummy caption!', ['dummy', 'caption', '!'], [1, 2, 3]




