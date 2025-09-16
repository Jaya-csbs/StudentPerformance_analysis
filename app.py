import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF

# Load trained model
model = joblib.load("model/student_perf_model.pkl")

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("🎓 Student Performance Predictor & Dashboard")

# ---------------- Sidebar Inputs ----------------
with st.sidebar:
    st.header("Enter Student Details")
    input_type = st.radio("Select Input Type", ["Single Student", "CSV Upload"])
    
    if input_type == "Single Student":
        gender = st.selectbox("Gender", ["female", "male"])
        race = st.selectbox("Race/Ethnicity", ["group A","group B","group C","group D","group E"])
        parent_edu = st.selectbox("Parental Level of Education", [
            "some high school","high school","some college","associate's degree","bachelor's degree","master's degree"])
        lunch = st.selectbox("Lunch Type", ["standard","free/reduced"])
        test_prep = st.selectbox("Test Prep Course", ["none","completed"])
        predict_button = st.button("Predict Performance")
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        predict_button = st.button("Predict for All Students")

# ---------------- Mappings ----------------
gender_map = {"female":0,"male":1}
race_map = {"group A":0,"group B":1,"group C":2,"group D":3,"group E":4}
parent_map = {"some high school":0,"high school":1,"some college":2,
              "associate's degree":3,"bachelor's degree":4,"master's degree":5}
lunch_map = {"free/reduced":0,"standard":1}
test_prep_map = {"none":0,"completed":1}

label_map = {0: "Low", 1: "Medium", 2: "High"}
color_map = {"Low":"🔴 Low", "Medium":"🟠 Medium", "High":"🟢 High"}

subject_tips_streamlit = {
    "Math": {"Low":"📘 Focus on solving more math problems.","Medium":"📗 Good! Try advanced exercises.","High":"🏆 Excellent! Try competitions."},
    "Reading": {"Low":"📖 Read daily 20 min.","Medium":"📚 Improve with diverse genres.","High":"🌟 Great! Explore advanced literature."},
    "Writing": {"Low":"✍️ Practice short essays.","Medium":"📝 Work on structure & vocab.","High":"🏅 Excellent! Try competitions."}
}

subject_tips_pdf = {
    "Math": {"Low":"Focus on solving more math problems.","Medium":"Good! Try advanced exercises.","High":"Excellent! Try competitions."},
    "Reading": {"Low":"Read daily 20 min.","Medium":"Improve with diverse genres.","High":"Great! Explore advanced literature."},
    "Writing": {"Low":"Practice short essays.","Medium":"Work on structure & vocab.","High":"Excellent! Try competitions."}
}

subjects = ["Math","Reading","Writing"]

# ---------------- Helper Functions ----------------
def preprocess_single(gender,race,parent_edu,lunch,test_prep):
    return np.array([[gender_map[gender], race_map[race], parent_map[parent_edu],
                      lunch_map[lunch], test_prep_map[test_prep]]])

def display_results(prediction):
    scores = prediction
    st.subheader("🎯 Predicted Performance Levels")
    for sub, score in zip(subjects, scores):
        score_label = label_map[int(score)]
        percent = 50 if score_label=="Low" else 75 if score_label=="Medium" else 100
        st.progress(percent/100)
        st.write(f"{sub}: {color_map[score_label]}")
    
    # Bar chart
    plt.figure(figsize=(6,4))
    plt.bar(subjects, [50 if label_map[int(x)]=="Low" else 75 if label_map[int(x)]=="Medium" else 100 for x in scores],
            color=['red','orange','green'])
    plt.ylim(0,100)
    plt.ylabel("Performance Score")
    st.pyplot(plt)

    # Radar chart
    fig = plt.figure(figsize=(5,5))
    angles = np.linspace(0, 2 * np.pi, len(subjects), endpoint=False).tolist()
    scores_percent = [50 if label_map[int(x)]=="Low" else 75 if label_map[int(x)]=="Medium" else 100 for x in scores]
    scores_percent += scores_percent[:1]
    angles += angles[:1]
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, scores_percent, 'o-', linewidth=2)
    ax.fill(angles, scores_percent, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), subjects)
    ax.set_ylim(0,100)
    st.subheader("📊 Radar Chart")
    st.pyplot(fig)

    # Suggestions
    st.subheader("💡 Subject-Specific Suggestions")
    for sub, score in zip(subjects, scores):
        score_label = label_map[int(score)]
        st.write(f"{sub}: {subject_tips_streamlit[sub][score_label]}")

# ---------------- Prediction & PDF ----------------
if predict_button:
    if input_type=="Single Student":
        features = preprocess_single(gender,race,parent_edu,lunch,test_prep)
        prediction = model.predict(features)[0]
        display_results(prediction)

        # PDF report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Student Performance Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        pdf.cell(0, 10, f"Gender: {gender}", ln=True)
        pdf.cell(0, 10, f"Race/Ethnicity: {race}", ln=True)
        pdf.cell(0, 10, f"Parental Education: {parent_edu}", ln=True)
        pdf.cell(0, 10, f"Lunch Type: {lunch}", ln=True)
        pdf.cell(0, 10, f"Test Prep Course: {test_prep}", ln=True)
        pdf.ln(10)
        pdf.cell(0, 10, "Predicted Performance:", ln=True)
        for sub, score in zip(subjects, prediction):
            score_label = label_map[int(score)]
            pdf.cell(0, 10, f"{sub}: {score_label}", ln=True)
            pdf.cell(0, 10, f"Suggestion: {subject_tips_pdf[sub][score_label]}", ln=True)
        pdf.output("Student_Performance_Report.pdf")
        st.success("✅ PDF report generated: Student_Performance_Report.pdf")

    else:  # CSV batch
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Rename columns from original dataset to short names
            df = df.rename(columns={
                "race/ethnicity": "race",
                "parental level of education": "parent_edu",
                "test preparation course": "test_prep"
            })

            # Convert to numeric using mappings
            feature_array = df[['gender','race','parent_edu','lunch','test_prep']].replace({
                'gender': gender_map,
                'race': race_map,
                'parent_edu': parent_map,
                'lunch': lunch_map,
                'test_prep': test_prep_map
            }).to_numpy()

            predictions = model.predict(feature_array)
            results = []
            for i, row in enumerate(df.itertuples()):
                scores = predictions[i]
                result = {"Math": label_map[int(scores[0])],
                          "Reading": label_map[int(scores[1])],
                          "Writing": label_map[int(scores[2])]}
                results.append(result)
            results_df = pd.DataFrame(results)
            final_df = pd.concat([df, results_df], axis=1)
            st.dataframe(final_df)
            final_df.to_csv("Batch_Predictions.csv", index=False)
            st.success("✅ CSV with predictions saved: Batch_Predictions.csv")
        else:
            st.warning("Please upload a CSV file!")
