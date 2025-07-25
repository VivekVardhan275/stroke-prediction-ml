import streamlit as st
import pickle
import numpy as np
import json
import requests 
models = {
    "Logistic Regression": "notebooks/models/logistic_model.pkl",
    "SVM": "notebooks/models/svc_model.pkl",
    "Naive Bayes": "notebooks/models/naive_bayes_model.pkl",
    "KNN": "notebooks/models/knn_model.pkl",
    "Random Forest": "notebooks/models/random_forest_model.pkl",
    "XGBoost": "notebooks/models/xgboost_model.pkl"
}
@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def preprocess_input(gender, age, hypertension, heart_disease,
                     avg_glucose_level, bmi, smoking_status):
    hypertension_val = 1 if hypertension == 'Yes' else 0
    heart_disease_val = 1 if heart_disease == 'Yes' else 0
    scalar = pickle.load(open("notebooks/models/scaler.pkl", "rb"))
    gender_le = pickle.load(open("notebooks/models/gender_le.pkl", "rb"))
    smoking_status_le = pickle.load(open("notebooks/models/smoking_status_le.pkl", "rb"))
    gender_encoded = gender_le.transform([gender])[0]
    smoking_status_encoded = smoking_status_le.transform([smoking_status])[0]
    numerical_array = np.array([[age, hypertension_val, heart_disease_val, avg_glucose_level, bmi]])
    scaled_numerical = scalar.transform(numerical_array)[0]
    input_array = np.array([[
    gender_encoded,
    scaled_numerical[0], # scaled age
    scaled_numerical[1], # scaled hypertension
    scaled_numerical[2], # scaled heart_disease
    scaled_numerical[3], # scaled avg_glucose_level
    scaled_numerical[4], # scaled bmi
    smoking_status_encoded
    ]])

    return input_array

def get_gemini_recommendations(user_data, prediction_result):
    """
    Calls the Gemini AI API to generate personalized health recommendations.
    The prompt is dynamically created based on the user's input and the stroke prediction.
    """
    api_key = "AIzaSyASjNCk1oxREML4AFhcx5joggaSyvDVj9Q"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    gender, age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status = user_data
    if prediction_result == 1:
        prompt_text = (
            f"Based on the following patient data and a prediction of HIGH STROKE RISK, "
            f"provide detailed, actionable health recommendations focused on stroke prevention. "
            f"The recommendations should be clear, concise, use bullet points, and strongly encourage consultation with a healthcare professional. "
            f"Data: Gender: {gender}, Age: {age}, Hypertension: {hypertension}, "
            f"Heart Disease: {heart_disease}, Avg Glucose: {avg_glucose_level}, BMI: {bmi}, "
            f"Smoking Status: {smoking_status}."
        )
    else:
        prompt_text = (
            f"Based on the following patient data and a prediction of LOW STROKE RISK, "
            f"provide general health maintenance tips to continue preventing stroke and promote overall well-being. "
            f"The tips should be clear, concise, use bullet points, and encourage regular check-ups and a proactive approach to health. "
            f"Data: Gender: {gender}, Age: {age}, Hypertension: {hypertension}, "
            f"Heart Disease: {heart_disease}, Avg Glucose: {avg_glucose_level}, BMI: {bmi}, "
            f"Smoking Status: {smoking_status}."
        )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_text}]
            }
        ]
    }

    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() 

        result = response.json()

        if result.get('candidates') and result['candidates'][0].get('content') and \
           result['candidates'][0]['content'].get('parts') and \
           result['candidates'][0]['content']['parts'][0].get('text'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Could not generate personalized tips. The AI response was malformed or empty."
    except requests.exceptions.RequestException as e:
        return f"Failed to get personalized tips due to an API connection error: {e}. Please check your internet connection or try again later."
    except json.JSONDecodeError:
        return "Failed to parse API response. Invalid JSON received."
    except Exception as e:
        return f"An unexpected error occurred while generating tips: {e}. Please report this issue."


st.title("🫀 Stroke Prediction App")
st.markdown("Predict the likelihood of a patient having a stroke using classical ML models.")
st.markdown('''
In our study, Logistic Regression emerged as the most effective and reliable model for stroke prediction,
outperforming more complex algorithms such as SVM, k-NN, Naïve Bayes, Random Forest, and XGBoost.
The dataset used was highly imbalanced, with 42,617 "No Stroke" cases (98.19%) and only 783 "Stroke" cases (1.81%)
out of 43,400 total records. This severe imbalance posed a significant challenge for most classifiers,
many of which defaulted to predicting only the majority class, thus failing to identify actual stroke cases.
Despite applying class balancing techniques like SMOTE, these models still struggled to generalize effectively.
In contrast, Logistic Regression maintained a strong balance between sensitivity (recall) and specificity,
accurately detecting minority class instances without compromising performance on the majority class.
Its robustness, simplicity, and interpretability make it particularly suitable for clinical applications
where early and accurate detection of stroke risk is critical.
Therefore, we recommend Logistic Regression as the preferred model for practical stroke prediction tasks.
''')

selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = load_model(models[selected_model_name])

st.subheader("Patient Health Information")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 100, 30)
hypertension = st.selectbox("Hypertension", ['No', 'Yes'])
heart_disease = st.selectbox("Heart Disease", ['No', 'Yes'])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=55.0, max_value=300.0, value=55.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=10.0)
smoking_status = st.selectbox("Smoking Status", ["non-smoker", "smoker"])

user_input_for_gemini = [gender, age, hypertension, heart_disease,
                         avg_glucose_level, bmi, smoking_status]

if st.button("Predict Stroke Risk"):
    input_data_for_model = preprocess_input(gender, age, hypertension, heart_disease,
                                            avg_glucose_level, bmi, smoking_status)
    prediction = model.predict(input_data_for_model)[0]
    if prediction == 1:
        st.warning("The model predicts that you may be at risk for stroke.")
    else:
        st.success("The model predicts you are not at immediate risk of stroke.")
    st.markdown("---") 
    st.markdown("Personalized Health Recommendations")
    with st.spinner("Generating personalized health tips..."):
        gemini_tips = get_gemini_recommendations(user_input_for_gemini, prediction)
        st.markdown(gemini_tips)
    st.markdown("---") 
    st.markdown(
        "Please remember that this app provides a predictive estimate and general health tips. "
        "Always consult with a qualified healthcare professional for medical advice and diagnosis."
    )
