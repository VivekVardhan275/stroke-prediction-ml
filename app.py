import streamlit as st
import pickle
import numpy as np
# Load models
models = {
    "Logistic Regression": "models/logistic_model.pkl",
    "SVM": "models/svc_model.pkl",
    "Naive Bayes": "models/naive_bayes_model.pkl",
    "KNN": "models/knn_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "XGBoost": "models/xgboost_model.pkl"
}

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    

def preprocess_input(gender, age, hypertension, heart_disease,
                     ever_married, work_type, Residence_type,
                     avg_glucose_level, bmi, smoking_status):

    # Manual encoding (use same as model training)
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    ever_married_map = {"No": 0, "Yes": 1}
    work_type_map = {"Private": 2, "Self-employed": 3, "Govt_job": 0, "children": 4, "Never_worked": 1}
    Residence_type_map = {"Urban": 1, "Rural": 0}
    smoking_map = {"never smoked": 2, "formerly smoked": 0, "smokes": 3, "Unknown": 1}

    input_array = np.array([[gender_map[gender], age, hypertension, heart_disease,
                             ever_married_map[ever_married], work_type_map[work_type],
                             Residence_type_map[Residence_type], avg_glucose_level,
                             bmi, smoking_map[smoking_status]]])

    return input_array


st.title("🧠 Stroke Prediction App")
st.markdown("Predict the likelihood of a patient having a stroke using classical ML models.")

# Sidebar to choose model
selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = load_model(models[selected_model_name])

# Input fields
st.subheader("Patient Health Information")
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", 1, 100, 30)
hypertension = st.selectbox("Hypertension", ['No', 'Yes'])
heart_disease = st.selectbox("Heart Disease", ['No', 'Yes'])
ever_married = st.selectbox("Ever Married", ['No', 'Yes'])
work_type = st.selectbox("Work Type", ["Private", "Self Employed", "Government Job", "Children", "Never Worked"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=55.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=100, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["non-smoker", "smoker"])

# Predict button
if st.button("Predict Stroke Risk"):
    input_data = preprocess_input(gender, age, hypertension, heart_disease,
                                   ever_married, work_type, Residence_type,
                                   avg_glucose_level, bmi, smoking_status)

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke! ({'Probability: {:.2f}'.format(prob) if prob else 'Model output: 1'})")
    else:
        st.success(f"✅ Low Risk of Stroke. ({'Probability: {:.2f}'.format(prob) if prob else 'Model output: 0'})")
