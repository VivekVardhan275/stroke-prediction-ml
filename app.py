import streamlit as st
import pickle
import numpy as np
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
                     ever_married, work_type,
                     avg_glucose_level, bmi, smoking_status):
    hypertension_val = 1 if hypertension == 'Yes' else 0
    heart_disease_val = 1 if heart_disease == 'Yes' else 0
    scalar = pickle.load(open("notebooks/models/scaler.pkl", "rb"))
    gender_le = pickle.load(open("notebooks/models/gender_le.pkl", "rb"))
    ever_married_le = pickle.load(open("notebooks/models/ever_married_le.pkl", "rb"))
    work_type_le = pickle.load(open("notebooks/models/work_type_le.pkl", "rb"))
    smoking_status_le = pickle.load(open("notebooks/models/smoking_status_le.pkl", "rb"))
    gender_encoded = gender_le.transform([gender])[0]
    ever_married_encoded = ever_married_le.transform([ever_married])[0]
    work_type_encoded = work_type_le.transform([work_type])[0]
    smoking_status_encoded = smoking_status_le.transform([smoking_status])[0]
    numerical_array = np.array([[age, hypertension_val, heart_disease_val, avg_glucose_level, bmi]])
    scaled_numerical = scalar.transform(numerical_array)[0]
    input_array = np.array([[
        gender_encoded,
        scaled_numerical[0],  # scaled age
        scaled_numerical[1],  # scaled hypertension
        scaled_numerical[2],  # scaled heart_disease
        ever_married_encoded,
        work_type_encoded,
        scaled_numerical[3],  # scaled avg_glucose_level
        scaled_numerical[4],  # scaled bmi
        smoking_status_encoded
    ]])

    return input_array

st.title("🧠 Stroke Prediction App")
st.markdown("Predict the likelihood of a patient having a stroke using classical ML models.")
st.markdown('''In our experiments, Logistic Regression consistently outperformed other classical machine learning models in correctly identifying both stroke and non-stroke cases. While more complex models like SVM, k-NN, Naïve Bayes, Random Forest, and XGBoost tended to predict only the dominant class (i.e., "No Stroke") — likely due to the inherent class imbalance — Logistic Regression demonstrated the best balance between sensitivity (recall) and specificity, especially after handling imbalance with techniques like SMOTE. Hence we recommend using Logistic Regression for stroke prediction tasks.''')
selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = load_model(models[selected_model_name])

st.subheader("Patient Health Information")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 100, 30)
hypertension = st.selectbox("Hypertension", ['No', 'Yes'])
heart_disease = st.selectbox("Heart Disease", ['No', 'Yes'])
ever_married = st.selectbox("Ever Married", ['No', 'Yes'])
work_type = st.selectbox("Work Type", ["Private", "Self Employed", "Government Job", "Children", "Never Worked"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=55.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["non-smoker", "smoker"])

if st.button("Predict Stroke Risk"):
    input_data = preprocess_input(gender, age, hypertension, heart_disease,
                                   ever_married, work_type,
                                   avg_glucose_level, bmi, smoking_status)

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke! ({'Probability: {:.2f}'.format(prob) if prob else 'Model output: 1'})")
    else:
        st.success(f"✅ Low Risk of Stroke. ({'Probability: {:.2f}'.format(prob) if prob else 'Model output: 0'})")
