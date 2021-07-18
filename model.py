import os, sys
import numpy as np
from PIL import Image

import jax
from transformers import ViTFeatureExtractor
from transformers import GPT2Tokenizer

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

# Main model -  ViTGPT2LM
from vit_gpt2.modeling_flax_vit_gpt2_lm import FlaxViTGPT2LMForConditionalGeneration

model_name_or_path = 'flax-community/vit-gpt2/checkpoints/ckpt_5/'
flax_vit_gpt2_lm = FlaxViTGPT2LMForConditionalGeneration.from_pretrained(model_name_or_path)

def predict(image):
    return 'dummy caption!', ['dummy', 'caption', '!'], [1, 2, 3]




