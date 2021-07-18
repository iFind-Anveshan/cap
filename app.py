import streamlit as st
from PIL import Image
import numpy as np


# Designing the interface
st.title("🖼️ French Image Caption App")

st.markdown(
    """
    An image caption model [ViT-GPT2](https://huggingface.co/flax-community/vit-gpt2/tree/main) by combining the ViT model and a French GPT2 model.
    [Part of the [Huggingface JAX/Flax event](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/).]\n
    The pretained weights of both models are loaded, with a set of randomly initialized cross-attention weigths.
    The model is trained on 65000 images from the COCO dataset for about 1500 steps, with the original english cpationis are translated to french for training purpose.
    """
)

#image = Image.open('samples/val_000000039769.jpg')
#show = st.image(image, use_column_width=True)
#show.image(image, 'Preloaded Image', use_column_width=True)

with st.spinner('Loading and compiling ViT-GPT2 model ...'):

    from model import *
    st.sidebar.write(f'Vit-GPT2 model loaded :)')

st.sidebar.title("Select a sample image")

sample_name = st.sidebar.selectbox(
    "Please choose an image",
    sample_fns
)

sample_name = f"COCO_val2014_{sample_name.replace('.jpg', '').zfill(12)}.jpg"
sample_path = os.path.join(sample_dir, sample_name)

image = Image.open(sample_path)
show = st.image(image, use_column_width=True)
show.image(image, '\n\nSelected Image', use_column_width=True)

# For newline
st.sidebar.write('\n')

with st.spinner('Generating image caption ...'):

    caption = predict(image)
    
    caption_en = translator.translate(caption, src='fr', dest='en').text
    st.header(f'**Prediction (in French) **{caption}')
    st.header(f'**English Translation**: {caption_en}')

st.sidebar.header("ViT-GPT2 predicts:")
st.sidebar.write(f"French: {caption}")
st.sidebar.write(f"English: {caption_en}")

show = st.sidebar.image(image, use_column_width=True)
show.image(image, use_column_width=True)

image.close()