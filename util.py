import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def classify(image, model_xray, class_names):
   
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

   
    image_array = np.asarray(image)

   
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

 
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array


    prediction = model_xray.predict(data)
    print("Prediction",prediction)
    print(prediction.shape)
   
    index = 0 if prediction[0][0] > 0.5 else 1  
    class_name = class_names[index]
    confidence_score = prediction[0]
    print("Class name:", class_name)
    print("confidence score:", confidence_score)

    return class_name, confidence_score