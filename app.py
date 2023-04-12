import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Define function to load and preprocess image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# Load saved CNN model
model = tf.keras.models.load_model('Xception_model_nonTrainable.h5')

# Define Streamlit app
def app():
    # Set app title
    st.title('CNN Image Classifier')
    st.write("Supported Plants")
    st.write("1)Blackboard tree          2)Arjun        3)Chinar       4)Gauva")
    st.write("5)Jamun        6)Jatropha         7)Lemon        8)Manago")
    st.write("9)Pomegranate            10)Pongame oiltree")

    # Create file uploader
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    cat = {0:'Blackboard tree_diseased', 1:'Blackboard tree_healthy',
       2:'Arjun_diseased', 3:'Arjun_healthy', 4:'Chinar_diseased',
       5:'Chinar_healthy', 6:'Gauva_diseased', 7:'Gauva_healthy',
       8:'Jamun_diseased', 9:'Jamun_healthy', 10:'Jatropha_diseased',
       11:'Jatropha_healthy', 12:'Lemon_diseased', 13:'Lemon_healthy',
       14:'Mango_diseased', 15:'Mango_healthy', 16:'Pomegranate_diseased',
       17:'Pomegranate_healthy', 18:'Pongame oiltree_diseased',
       19:'Pongame oiltree_healthy'}
    # Make prediction when file is uploaded
    if uploaded_file is not None:
        # Preprocess uploaded image
        img = preprocess_image(uploaded_file)

        # Make prediction
        predictions = model.predict(img[np.newaxis, ...])
        pred = np.argmax(predictions)
        f_pred = cat[pred]
        # Display prediction
        st.write('Prediction:')
        st.write(f_pred)

# Run app
if __name__ == '__main__':
    app()