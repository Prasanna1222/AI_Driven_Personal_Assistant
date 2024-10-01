import streamlit as st
import pickle
from gtts import gTTS
import numpy as np
import os
import pandas as pd
from fuzzywuzzy import process
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import time
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import pytesseract
import cv2
from keras.models import load_model
from util import classify
import spacy
from transformers import pipeline
import io
import mlflow
import mlflow.keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 



model = pickle.load(open('voting_classifiers.pkl', 'rb'))



precautions_df = pd.read_csv('data/precautions_df.csv')
diets_df = pd.read_csv('data/diets.csv')
medications_df = pd.read_csv('data/medications.csv')
workout_df = pd.read_csv('data/workout_df.csv')
description_df = pd.read_csv('data/description.csv')


st.set_page_config(
    page_title="Medical Recommendation System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


with st.sidebar:
    page = option_menu("BAYMAX", ["Home", "Symptoms", "Lab Report", "X-Ray Report"],
                       icons=['house', 'heart-pulse', 'file-earmark-medical', 'lungs'], 
                       menu_icon="cast", default_index=0)




stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_model1():
    return pickle.load(open('voting_classifiers.pkl', 'rb'))

def normalize_symptom(symptom):
    
    tokens = word_tokenize(symptom.lower())
    tokens = [word for word in tokens if word not in stop_words]  
    stemmed_tokens = [stemmer.stem(word) for word in tokens]  
    normalized_symptom = ' '.join(stemmed_tokens)
    standardized_symptom = process.extractOne(normalized_symptom, list(symptoms_dict.keys()))
    return standardized_symptom[0] if standardized_symptom else symptom


model = load_model1()

def get_predicted_value(patient_symptoms):
    
    normalized_symptoms = [normalize_symptom(symptom) for symptom in patient_symptoms]
    
    input_vector = np.zeros(len(symptoms_dict))
    for item in normalized_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
            
    return diseases_list[model.predict([input_vector])[0]]

def helper(disease):
    desc = description_df[description_df['Disease'] == disease]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions_df[precautions_df['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications_df[medications_df['Disease'] == disease]['Medication']
    med = [med for med in med.values]

    diet = diets_df[diets_df['Disease'] == disease]['Diet']
    diet = [diet for diet in diet.values]

    wrkout = workout_df[workout_df['disease'] == disease]['workout']

    return desc, pre, med, diet, wrkout
def show_custom_spinner():
   
    spinner_html = """
    <div class="wrapper">
        <div class="circle"></div>
        <div class="circle"></div>
        <div class="circle"></div>
        <div class="shadow"></div>
        <div class="shadow"></div>
        <div class="shadow"></div>
    </div>
    <style>
    .wrapper {
      width: 200px;
      height: 60px;
      position: relative;
      z-index: 1;
    }
    .circle {
      width: 20px;
      height: 20px;
      position: absolute;
      border-radius: 50%;
      background-color: #fff;
      left: 15%;
      transform-origin: 50%;
      animation: circle7124 .5s alternate infinite ease;
    }
    @keyframes circle7124 {
      0% {
        top: 60px;
        height: 5px;
        border-radius: 50px 50px 25px 25px;
        transform: scaleX(1.7);
      }
      40% {
        height: 20px;
        border-radius: 50%;
        transform: scaleX(1);
      }
      100% {
        top: 0%;
      }
    }
    .circle:nth-child(2) {
      left: 45%;
      animation-delay: .2s;
    }
    .circle:nth-child(3) {
      left: auto;
      right: 15%;
      animation-delay: .3s;
    }
    .shadow {
      width: 20px;
      height: 4px;
      border-radius: 50%;
      background-color: rgba(0,0,0,0.9);
      position: absolute;
      top: 62px;
      transform-origin: 50%;
      z-index: -1;
      left: 15%;
      filter: blur(1px);
      animation: shadow046 .5s alternate infinite ease;
    }
    @keyframes shadow046 {
      0% {
        transform: scaleX(1.5);
      }
      40% {
        transform: scaleX(1);
        opacity: .7;
      }
      100% {
        transform: scaleX(.2);
        opacity: .4;
      }
    }
    .shadow:nth-child(4) {
      left: 45%;
      animation-delay: .2s
    }
    .shadow:nth-child(5) {
      left: auto;
      right: 15%;
      animation-delay: .3s;
    }
    </style>
    """
    return spinner_html
streamlit_style = """
			<style>
			@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Poppins', sans-serif;
			}
			"""
st.write(streamlit_style, unsafe_allow_html=True)



symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 27: 'Hyperthyroidism', 2: 'Arthritis', 5: 'Psoriasis', 25: 'Impetigo'}


nlp_ner = spacy.load('en_core_web_sm')


def preprocess_image(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    processed_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]  
    return processed_image


def extract_key_information(text):
    doc = nlp_ner(text)
    key_info = []
    
    
    for ent in doc.ents:
        if ent.label_ in ['DATE', 'ORG', 'PERSON', 'QUANTITY', 'MONEY', 'GPE']:
            key_info.append((ent.text, ent.label_))
    
    return key_info


def summarize_text(text):
    
    text = text[:1000] if len(text) > 1000 else text
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']





if page == "Home":
    st.title("Meet BAYMAX - Your Medical Assistant")
    st.markdown(
        """
        <div style="color: #00E676; font-size: 24px; font-weight: bold;">
        BAYMAX is designed to assist you with your medical needs through various advanced functionalities.
        </div>
        <div style="color: #03A9F4; font-size: 18px; margin-top: 20px;">
        Our system leverages state-of-the-art technology to provide:
        </div>
        <ul style="color: #FFAB00; font-size: 16px; list-style-type: disc; margin-left: 20px;">
            <li><strong>Symptom-based Prediction</strong>: Enter your symptoms to receive a potential diagnosis and tailored recommendations.</li>
            <li><strong>OCR-based Lab Report Analysis</strong>: Upload an image of your lab report to extract and analyze critical information.</li>
            <li><strong>Pneumonia Detection from X-Ray</strong>: Upload a chest X-ray image to detect pneumonia using advanced image classification.</li>
            <li><strong>Text-to-Speech Diagnosis</strong>: Listen to the summary of your diagnosis and recommendations with our integrated text-to-speech feature.</li>
        </ul>
        <div style="color: #D500F9; font-size: 16px; margin-top: 20px;">
        Explore each feature using the sidebar to get personalized health insights and recommendations.
        </div>
        """, unsafe_allow_html=True
    )


if page == "Symptoms":
    st.title("Symptom-based Medical Prediction")

    symptoms_input = st.text_input("Enter Symptoms (comma-separated):")

    if st.button("Predict"):
        if symptoms_input:
            symptoms = [s.strip() for s in symptoms_input.split(',')]
            normalized_symptoms = [normalize_symptom(symptom) for symptom in symptoms]
                      
            st.write("**Normalized Symptoms:**")
            st.write(", ".join(normalized_symptoms))
            predicted_disease = get_predicted_value(symptoms)
           
            spinner_placeholder = st.empty()
        
            spinner_placeholder.markdown(show_custom_spinner(), unsafe_allow_html=True)
        
            time.sleep(2)
            spinner_placeholder.empty()
        
            st.success(f"Predicted Disease: {predicted_disease}")
            
            st.write(f"**Recommendations for {predicted_disease}:**")
                
            desc, precautions, medications, rec_diet, workout = helper(predicted_disease)
                
            st.write("**Description**:")
            st.write(desc)
                
            if len(precautions) > 0:
                st.write("**Precautions**:")
                for pred in precautions:
                    st.write(pred)
                    
            if len(medications) > 0:
                st.write("**Medications**:")
                for med in medications:
                        st.write(med)
                    
            if len(rec_diet) > 0:
                st.write("**Dietary Suggestions**:")
                for diet in rec_diet:
                        st.write(diet)
                    
            if len(workout) > 0:
                st.write("**Workout Suggestions**:")
                st.write(workout)

               
      
    
        else:
            st.error("Please enter symptoms.")


if page == "Lab Report":
    model_name = "sshleifer/distilbart-cnn-12-6"
    summarizer = pipeline("summarization", model=model_name)
    st.title("OCR Lab Report Analysis")
    st.write("Upload an image file to extract text using OCR.")

    uploaded_file = st.file_uploader("Choose an image file...", type=['png', 'jpg', 'jpeg', 'bmp'])

    if uploaded_file is not None:
       
        try:
            
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption='Uploaded Image.', use_column_width=True)

            
            processed_image = preprocess_image(img)
          
            
            
            processed_image_pil = Image.fromarray(processed_image)
          

            
            text = pytesseract.image_to_string(processed_image)
            st.write("Extracted Text:")
            st.text_area("OCR Output", value=text, height=250)

            
            key_info = extract_key_information(text)
            st.write("Key Information Extracted:")
            for info in key_info:
                st.write(f"{info[0]} ({info[1]})")

           
            summary = summarize_text(text)
            st.write("Summary of Extracted Text:")
            st.text_area("Text Summary", value=summary, height=150)
            if st.button("Read Summary"):
              try:
        
                tts = gTTS(text=summary, lang='en')
        
       
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0) 
        
        
                st.audio(audio_buffer, format="audio/mp3")
              except Exception as e:
                  st.error(f"Error playing audio:{e}")

        except Exception as e:
          st.error(f"Error processing file: {e}")
if page == "X-Ray Report":
    st.title('Pneumonia classification')


    st.header('Please upload a chest X-ray image')


    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])


   
    model_xray = load_model('xray_classifier.h5')
    model_xray.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
      


    with open('./model/labels.txt', 'r') as f:
      class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
      f.close()


    if file is not None:
      image = Image.open(file).convert('RGB')
      

    
      class_name, conf_score = classify(image, model_xray, class_names)
      spinner_placeholder = st.empty()
        # Show the custom spinner
      spinner_placeholder.markdown(show_custom_spinner(), unsafe_allow_html=True)
        
       
      time.sleep(2)
      spinner_placeholder.empty()
      st.image(image, use_column_width=True)


      if class_name == "NORMAL":
        st.success("## Test Result: {}".format(class_name))
        st.write("### score: {:.1f}%".format(int(conf_score * 100)))
        st.write("The pneumonia test result is normal. No signs of pneumonia detected.")
      else:
        st.write("## Test Result: {}".format(class_name))
        st.write("The pneumonia test result indicates pneumonia. Please seek medical advice immediately.")
        st.write("### score: {:.1f}%".format(int(conf_score * 100)))
