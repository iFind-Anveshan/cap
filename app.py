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

st.sidebar.title("Upload Image")

# Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
# Choose your own image
uploaded_file = st.sidebar.file_uploader(" ", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    show = st.image(image, use_column_width=True)
    show.image(image, 'Uploaded Image', use_column_width=True)


# For newline
st.sidebar.write('\n')

if st.sidebar.button("Click here to get image caption"):

    if uploaded_file is None:

        st.sidebar.write("Please upload an Image to Classify")

    else:

        with st.spinner('Generating image caption ...'):

            caption, tokens, token_ids = predict(image)

            st.success(f'caption: {caption}')
            st.success(f'tokens: {tokens}')
            st.success(f'token ids: {token_ids}')

        st.sidebar.header("ViT-GPT2 predicts:")
        st.sidebar.write(f"caption: {caption}", '\n')
