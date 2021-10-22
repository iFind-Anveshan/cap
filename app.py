import streamlit as st
import requests


# Designing the interface
st.title("🖼️ Image Captioning Demo 📝")
st.write("[Yih-Dar SHIEH](https://huggingface.co/ydshieh)")

st.sidebar.markdown(
    """
    An image captioning model by combining ViT model with GPT2 model.
    The encoder (ViT) and decoder (GPT2) are combined using Hugging Face transformers' [Vision-To-Text Encoder-Decoder
    framework](https://huggingface.co/transformers/master/model_doc/visionencoderdecoder.html).
    The pretrained weights of both models are loaded, with a set of randomly initialized cross-attention weights.
    The model is trained on the COCO 2017 dataset for about 6900 steps (batch_size=256).
    [Follow-up work of [Huggingface JAX/Flax event](https://discuss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/).]\n
    """
)

with st.spinner('Loading and compiling ViT-GPT2 model ...'):
    from model import *


st.sidebar.title("Select a sample image")
image_id = st.sidebar.selectbox(
    "Please choose a sample image",
    sample_image_ids
)

random_image_id = None
if st.sidebar.button("Random COCO 2017 (val) images"):
    random_image_id = get_random_image_id()

if random_image_id is not None:
    image_id = random_image_id

st.write(image_id)

sample_name = f"COCO_val2017_{str(image_id).zfill(12)}.jpg"
sample_path = os.path.join(sample_dir, sample_name)

if os.path.isfile(sample_path):
    image = Image.open(sample_path)
else:
    url = f"http://images.cocodataset.org/val2017/{str(image_id).zfill(12)}.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

resized = image.resize(size=(384, 384))
show = st.image(resized, width=384)
show.image(resized, '\n\nSelected Image', width=384)
resized.close()

# For newline
st.sidebar.write('\n')

with st.spinner('Generating image caption ...'):

    caption = predict(image)

    caption_en = caption
    st.header(f'Predicted caption:\n\n')
    st.subheader(caption_en)

st.sidebar.header("ViT-GPT2 predicts:")
st.sidebar.write(f"**English**: {caption}")

image.close()
