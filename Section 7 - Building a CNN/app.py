import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

# Load your trained model (ensure it's in the same directory as this script or provide the full path)
model = tf.keras.models.load_model('model.h5')

def predict(image_data):
    """Function to predict the class of the image"""
    # Preprocess the image to fit your model's input requirements
    img = image_data.resize((64, 64))
    img = img.convert('RGB')
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make a prediction using your model
    result = model.predict(img)

    if result[0][0] > 0.5:
        return 'Dog', result[0][0]
    else:
        return 'Cat', 1 - result[0][0]

# Streamlit app layout
st.title("Cat vs Dog Image Classifier")
st.write("This application uses a convolutional neural network to classify images as either a cat or a dog.")

st.write("## How It Works")
st.write("""
- The goal of this CNN is to accurately classify dogs vs cats given an image.
- Step 1: Data Augmentation: I applied rescaling, sheering, zooming, and flipping of training images to increase dataset size and prevent model overfitting
- Step 2: I then resize the image to be 64x64 to ensure consistent input size
- Step 3: I initialized a Sequential model using Keras
- Step 4: I applied a convolutional layer with filter size of 32 and kernel size of 3x3, utilizing the relu activation function
- Step 5: I applied a max pooling layer with pool size of 2x2 and stride of 2
- Step 6: I repeated Step 4 and 5 to add a second convolutional and max pooling layer
- Step 7: I flattened the output of the second max pooling layer
- Step 8: I add a fully connected hidden layer
- Step 9: I added the final output layer with a sigmoid activation function to keep the output between range of 0 and 1, where I can calculate the probability of the input belonging to class labeled as '1' (which is dog in this case)
- After compiling and training the model, we have a working CNN!
- Try it out by uploading an image!
""")

st.write("## Try It Out")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image_to_show = Image.open(uploaded_file)
    st.image(image_to_show, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    prediction, probability = predict(image_to_show)
    
    st.write(f'The CNN prediction is {prediction} with {probability * 100:.2f}% probability')

