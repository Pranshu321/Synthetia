import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import io
import librosa
import librosa.display
import numpy as np
import os
import tempfile

# Function to preprocess audio data


def save_audio_to_file(uploadedfile):
    with open(os.path.join("AudioModel\\audio", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    # return the file path
    return os.path.join("AudioModel\\audio", uploadedfile.name)


def predict_the_output(model, audio_bytes, genre_mapping={0: "Fake", 1: "Real"}):
    file_path = save_audio_to_file(audio_bytes)
    signal, sample_rate = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T

    mfcc = np.resize(mfcc, (130, 13, 1))

    mfcc = mfcc[np.newaxis, ...]

    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction, axis=1)

    genre_label = genre_mapping[predicted_index[0]]

    return genre_label

# Function for audio prediction


def predict_audio():
    html_temp = """
    <div style="background-color:green;padding:10px;margin-bottom:20px;">
    <h2 style="color:white;text-align:center;">Deepfake Audio Detection</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("Upload an audio for classification as either original or synthetic")

    option = st.sidebar.selectbox(
        "Select a model", ["CNN", "LSTM"])

    # Load the appropriate model based on the selection
    if option == "CNN":
        # Make sure to use correct path
        model = load_model('AudioModel/cnn_audio.h5')
    elif option == "LSTM":
        # Make sure to use correct path
        model = load_model('AudioModel/lstm_audio.h5')

    uploaded_file = st.file_uploader(
        'Choose an audio to upload…', type=["mp3", "wav"])
    if uploaded_file is not None:
        # Read the file as bytes
        audio_bytes = uploaded_file.read()

        # Display the uploaded audio
        st.audio(audio_bytes, format='audio/wav', start_time=0)

        result = ""
        if st.button("Predict"):
            result = predict_the_output(
                audio_bytes=uploaded_file, model=model)
            st.success('The audio provided is '+result)
