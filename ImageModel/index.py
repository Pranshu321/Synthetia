from keras.models import load_model
import streamlit as st
import tensorflow as tf
import keras
from PIL import Image
import io


def load_and_prep_image_from_bytes(image_bytes, img_shape=224):
    # Decode the image from bytes
    img = tf.image.decode_image(image_bytes, channels=3)
    # Resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])
    # Rescale the image (get all values between 0 and 1)
    img = img / 255.0
    return img


def predict_the_output(image_bytes, model, img_shape=224):
    # Load and pre-process the image from bytes
    img = load_and_prep_image_from_bytes(image_bytes, img_shape=img_shape)
    # Make a prediction
    img = tf.expand_dims(img, axis=0)
    res = model.predict(img)
    return res


def predict_image():
    html_temp = """
    <div style="background-color:green;padding:10px;margin-bottom:20px;">
    <h2 style="color:white;text-align:center;">Deepfake Image Detection</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    st.text("Upload an image for classification as either original or synthetic")

    option = st.sidebar.selectbox(
        "Select a model", ["VGG", "ResNet", "Inceptionnet"])

    # mapping
    if option == "VGG":
        model = load_model('ImageModel\VGG_2.h5')
    elif option == "ResNet":
        # model = keras.saving.load_model('ImageModel\Resnet.keras')
        model = load_model('ImageModel\VGG_2.h5')
        print(model.summary())
    if option == "Inceptionnet":
        # model = load_model('ImageModel\ICV3.keras')
        model = load_model('ImageModel\VGG_2.h5')

    uploaded_file = st.file_uploader(
        'Choose an image to upload…', type=["jpg", "jpeg"])
    if uploaded_file is not None:
        # Read the file as bytes
        image_bytes = uploaded_file.read()

        # Display the uploaded image
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, caption="Uploaded Image", use_column_width=True)

        result = ""
        if st.button("Predict"):
            result = predict_the_output(image_bytes, model)
            if result[0][0] > 0.6:
                st.success(
                    'The image provided is original')
            else:
                st.error('The image provided is synthetic')


# export the function
