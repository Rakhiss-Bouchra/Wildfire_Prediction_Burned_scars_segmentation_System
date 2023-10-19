import streamlit as st
import numpy as np
import os
import requests
import PIL
from io import BytesIO
# from dotenv import load_dotenv


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K

def onehot_to_mask(onehot):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    output[single_layer == 0] = (0, 0, 0)
    output[single_layer == 1] = (128, 128, 128)
    return np.uint8(output)

def iou_coef(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
    return final_coef_value

# Predict
def predict(model, image):
    prediction = model.predict(image)
    response = "Wildfire Risk Percentage: {:.2f}%".format(100 * prediction[0][1])
    return response

def segment(model, image):
    prediction = model.predict(image)
    tmp_mask = onehot_to_mask(prediction[0])
    return tmp_mask

# Create the Streamlit app
def main():
    st.title("Wildfire System App")

    # Create tabs to select the model
    model_selection = st.radio(
        "Select a Model",
        ["Risk Prediction", "Burn Scars Segmentation"],
        key="model_selection",
    )

    if model_selection == "Risk Prediction":
        st.header("Input Coordinates")
        # Add input fields for longitude and latitude
        longitude = st.number_input("Longitude", value=0.0, step=0.0001, format="%.6f")
        latitude = st.number_input("Latitude", value=0.0, step=0.0001, format="%.6f")
    
    # Allow user to upload an image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Add a button to trigger the app
    button_clicked = st.button("Submit")

    # Wait for the button to be clicked
    if button_clicked:
        if model_selection == "Risk Prediction":
            target = (350, 350)
        else:
            target = (512, 512)

        if uploaded_file is not None:
            # Load and preprocess the image
            image = load_img(uploaded_file, target_size=target)
            
        elif uploaded_file is None and model_selection == "Risk Prediction":
            # load_dotenv()
            # Predict using coordinates
            # api_key = os.environ.get("api_key")
            #api_key = st.secrets["api_key"]
            api_key = "pk.eyJ1IjoiYWJkb3VhYWJhIiwiYSI6ImNsZGRkYW9wNjAyYTYzb3E3ZzAyZWlveGMifQ.R5G_lFqxAMZfuULGfM5qBQ"
            center = str(longitude) + ',' + str(latitude)
            rest = ',15,0/350x350?access_token=' + api_key + '&logo=false&attribution=false'
            url = 'https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/' + center + rest

            # load & preprocess the image so the model can treat it properly
            response = requests.get(url)
            image = PIL.Image.open(BytesIO(response.content))


        st.image(image, caption="Uploaded Image", use_column_width=True)
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = image.astype('float32')
        image = image / 255.0


        # Load the selected model
        if model_selection == "Risk Prediction":
            model = load_model("first_model.hdf5")
            result = predict(model, image)
            # Display the classification result
            st.write("Predicting...")
            st.write(result)
        else:
            model = load_model("burned-area-2classes-SegNet.h5")
            result = segment(model, image)
            # Display the images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(result, caption="Predicted Mask", use_column_width=True)
            with col2:
                blend = ((np.array(image[0]) * 255 * 0.5) + (result * 0.5)) / 255.0
                st.image(blend, caption="Blended Image", use_column_width=True)


if __name__ == "__main__":
    get_custom_objects()['iou_coef'] = iou_coef
    main()
