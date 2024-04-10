import streamlit as st
from ImageModel import index
from AudioModel import index as audio_index

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
