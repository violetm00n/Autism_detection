import streamlit as st
import pandas as pd
import numpy as np
import pickle

bootstrap = """
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
"""
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

svc_model = load_model('svc_model.pkl')
logreg_model = load_model('logreg_model.pkl')
xgb_model = load_model('xgb_model.pkl')
rf_model = load_model('rf_model.pkl')
st.title('Autism Detection')
input_data = {}
input_data['Social_Responsiveness_Scale']= st.slider("Has the person faced challenges or experiences related to social communication and interaction difficulties?", min_value=0, max_value=10, value=5)
input_data['Age_Years'] = st.number_input('What is your age in years?', value=0)
option = st.radio('Is there any Speech Delay or Language Disorder for that person?', ("Yes", "No"))
input_data['Speech Delay/Language Disorder'] = 1 if option == "Yes" else 0
option = st.radio('Does the person have a trouble learning', ("Yes", "No"))
input_data['Learning disorder'] = 1 if option == "Yes" else 0
option = st.radio('Does the person have any Genetic Disorders', ("Yes", "No"))
input_data['Genetic_Disorders'] = 1 if option == "Yes" else 0
option = st.radio('Did the person go through Depression', ("Yes", "No"))
input_data['Depression'] = 1 if option == "Yes" else 0
option = st.radio('Intellectual Disability', ("Yes", "No"))
input_data['Global developoental delay/intellectual disability'] = 1 if option == "Yes" else 0
option = st.radio('Behavioural Issues', ("Yes", "No"))
input_data['Social/Behavioural Issues'] = 1 if option == "Yes" else 0
input_data['Childhood Autism Rating Scale']= st.slider("Childhood Autism Rating Scale", min_value=0, max_value=4, value=2)
option = st.radio('Anxiety Disorder', ("Yes", "No"))
input_data['Anxiety_disorder'] = 1 if option == "Yes" else 0
option = st.radio('Sex', ("Male", "Female"))
input_data['Sex'] = 1 if option == "Male" else 0
option = st.radio('Jaundice', ("Yes", "No"))
input_data['Jaundice'] = 1 if option == "Yes" else 0
option = st.radio('Family member with ASD', ("Yes", "No"))
input_data['Family_member_with_ASD'] = 1 if option == "Yes" else 0
if st.button('Predict'):
    input_df = pd.DataFrame([input_data])
    svc_pred = svc_model.predict(input_df)[0]
    logreg_pred = logreg_model.predict(input_df)[0]
    xgb_pred = xgb_model.predict(input_df)[0]
    
    # Prepare ensemble input
    ensemble_input = pd.DataFrame({'SVC': [svc_pred], 'LogisticRegression': [logreg_pred], 'XGBoost': [xgb_pred]})
    final_pred = rf_model.predict(ensemble_input)[0]
    
    # Display the prediction results
    
    st.subheader('Autism Detection:')
    if svc_pred==1:
        st.write(f'Final Detection: The Person has autism')
    else:
        st.write("The Person doesn't have autism")
    


   