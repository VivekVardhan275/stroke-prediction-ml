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
        scaled_numerical[0],  # scaled age
        scaled_numerical[1],  # scaled hypertension
        scaled_numerical[2],  # scaled heart_disease
        scaled_numerical[3],  # scaled avg_glucose_level
        scaled_numerical[4],  # scaled bmi
        smoking_status_encoded
    ]])

    return input_array

st.title("🧠 Stroke Prediction App")
st.markdown("Predict the likelihood of a patient having a stroke using classical ML models.")
st.markdown('''In our study, Logistic Regression emerged as the most effective and reliable model for stroke prediction, outperforming more complex algorithms such as SVM, k-NN, Naïve Bayes, Random Forest, and XGBoost. The dataset used was highly imbalanced, with 42,617 "No Stroke" cases (98.19%) and only 783 "Stroke" cases (1.81%) out of 43,400 total records. This severe imbalance posed a significant challenge for most classifiers, many of which defaulted to predicting only the majority class, thus failing to identify actual stroke cases. Despite applying class balancing techniques like SMOTE, these models still struggled to generalize effectively. In contrast, Logistic Regression maintained a strong balance between sensitivity (recall) and specificity, accurately detecting minority class instances without compromising performance on the majority class. Its robustness, simplicity, and interpretability make it particularly suitable for clinical applications where early and accurate detection of stroke risk is critical. Therefore, we recommend Logistic Regression as the preferred model for practical stroke prediction tasks.''')
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
input = [gender, age, hypertension, heart_disease,
         avg_glucose_level, bmi, smoking_status]
def get_recommendations(user_input):
    recommendations = []

    # Unpack user inputs
    gender, age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status = user_input

    # Rule: High BP
    if hypertension == 'Yes':
        recommendations.append("🔴 Manage Hypertension: Monitor your blood pressure regularly and follow prescribed medication or lifestyle changes.")

    # Rule: Heart disease
    if heart_disease == 'Yes':
        recommendations.append("❤️ Heart Health: Having heart disease increases stroke risk. Regular cardiology checkups and lifestyle changes are crucial.")

    # Rule: High glucose
    if avg_glucose_level > 140:
        recommendations.append("🍬 Control Blood Sugar: Elevated glucose levels can damage blood vessels. Follow a diabetic-friendly diet and monitor sugar levels.")

    # Rule: High BMI
    if bmi > 25:
        recommendations.append("⚖️ Maintain a Healthy Weight: Your BMI indicates overweight or obesity. Adopting a healthy diet and exercise routine is important.")

    # Rule: Smoker
    if smoking_status.lower() in ['smoker']:
        recommendations.append("🚭 Quit Smoking: Smoking is a major stroke risk factor. Quitting improves your vascular health quickly.")

    # Rule: Age over 60
    if age >= 60:
        recommendations.append("🧓 Senior Care: Being over 60 increases stroke risk. Regular health checkups and early intervention are key.")


    if not recommendations:
        recommendations.append("✅ You currently show no major lifestyle risk indicators. Maintain a healthy routine and monitor key vitals regularly.")

    return recommendations

if st.button("Predict Stroke Risk"):
    input_data = preprocess_input(gender, age, hypertension, heart_disease,
                                   avg_glucose_level, bmi, smoking_status)

    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.warning("⚠️ The model predicts that you may be at risk for stroke.")
        
        st.markdown("### 🛡️ Personalized Health Recommendations")
        tips = get_recommendations(input)
        for tip in tips:
            st.markdown(f"- {tip}")

    else:
        st.success("✅ The model predicts you are not at immediate risk of stroke.")
        st.markdown("Keep maintaining a healthy lifestyle. Stay proactive with regular checkups!")

