from keras.models import load_model
import streamlit as st
import tensorflow as tf
from PIL import Image
from ImageModel import index
from AudioModel import index as audio_index
import io

# Load the model


def main():
    st.title("Deepfake Detection")

    option = st.sidebar.selectbox("Select type", ["Image", "Audio"])

    if option == "Image":
        index.predict_image()

    elif option == "Audio":
        audio_index.predict_audio()


if __name__ == '__main__':
    main()
