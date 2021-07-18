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
    (
        "sample 1",
        "sample 2",
        "sample 3",
        "sample 4"
    )
)

sample_name = f'sample_{sample_name.split()[-1].zfill(2)}.jpg'
sample_path = f'samples/{sample_name}'

image = Image.open(sample_path)
show = st.image(image, use_column_width=True)
show.image(image, 'Uploaded Image', use_column_width=True)


# For newline
st.sidebar.write('\n')

# if st.sidebar.button("Click here to get image caption"):

with st.spinner('Generating image caption ...'):

    caption, tokens, token_ids = predict_dummy(image)

    st.success(f'caption: {caption}')
    st.success(f'tokens: {tokens}')
    st.success(f'token ids: {token_ids}')

st.sidebar.header("ViT-GPT2 predicts:")
st.sidebar.write(f"caption: {caption}", '\n')
