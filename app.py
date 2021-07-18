import streamlit as st
from PIL import Image
import numpy as np


# Designing the interface
st.title("French Image Caption App")
# For newline
st.write('\n')

#image = Image.open('samples/val_000000039769.jpg')
#show = st.image(image, use_column_width=True)
#show.image(image, 'Preloaded Image', use_column_width=True)

with st.spinner('Loading ViT-GPT2 model ...'):

    from model import *
    st.sidebar.write(f'Vit-GPT2 model loaded :)')

st.sidebar.title("Select a sample image")

sample_name = st.sidebar.selectbox(
    "Please Choose the Model",
    sample_fns
)

sample_name = f'COCO_val2014_{sample_name.replace('.jpg', '').zfill(12)}.jpg'
sample_path = os.path.join(sample_dir, sample_name)

image = Image.open(sample_path)
show = st.image(image, use_column_width=True)
show.image(image, 'Selected Image', use_column_width=True)

# For newline
st.sidebar.write('\n')

with st.spinner('Generating image caption ...'):

    caption = predict_dummy(image)
    image.close()
    st.success(f'caption: {caption}')

st.sidebar.header("ViT-GPT2 predicts:")
st.sidebar.write(f"caption: {caption}", '\n')
