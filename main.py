import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="my_thumbnail.png", 
    layout="wide"
)

# import dataset
file_name = "augmented_balanced_dataset_m.csv"
df = pd.read_csv(file_name)

# fixing table for display
df.columns = df.columns.str.replace(".", " ")

# hardcoded importance data from cross-validation hehe
importance_data = {
    'Feature': [
        'Read Actions',
        'Days Logged In',
        'Quiz Actions',
        'Create Actions',
        'Forum Actions',
        'Submission Actions',
        'Update Actions',
        'Delete Actions'
    ],
    'Importance': [
        0.255351,
        0.184933,
        0.145841,
        0.120962,
        0.104755,
        0.094029,
        0.081678,
        0.012452
    ]
}
importance_df = pd.DataFrame(importance_data)

# load model
model_name = "MP2_random_forest.joblib"
model = joblib.load(model_name)

# load pipeline
package = joblib.load('random_forest_pipeline.joblib')
scaler = package['scaler']

# app
st.title("Student Performance Predictor")
# table = st.columns(2)
# st.write("### Student Attribute List")
# st.dataframe(df)
# st.write("")

# input fields
# st.write("### Predict Student Performance")

space1, input_col1, input_col2, space2 = st.columns(4)
# space1.write("")  
submission = space1.number_input("Number of Submissions", min_value=0, max_value=100, value=6, required=True)
quiz = space1.number_input("Number of Quizzes", min_value=0, max_value=100, value=43, required=True)
forum = input_col1.number_input("Number of Forum Participation", min_value=0, max_value=100, value=8, required=True)
create = input_col1.number_input("Number of Create Actions", min_value=0, max_value=100, value=10, required=True)

read = input_col2.number_input("Number of Read Actions", min_value=0, value=866, required=True)
update = input_col2.number_input("Number of Update Actions", min_value=0, value=16, required=True)
delete = space2.number_input("Number of Delete Actions", min_value=0, value=0, required=True)
days = space2.number_input("Days Logged In", min_value=0, max_value=365, value=15, required=True)

# prediction
sample = pd.DataFrame(
    {
        "Submission.Actions" : [submission],
         "Quiz.Actions": [quiz],
         "Forum.Actions": [forum],
         "Create.Actions": [create],
         "Read.Actions": [read],
         "Update.Actions": [update],
         "Delete.Actions": [delete],
         "Days.Logged.In": [days]
    }
)

# Apply the saved scale transformations
new_df_scaled = sample.copy()
new_df_scaled[sample.columns] = scaler.transform(sample[sample.columns])

# Prediction and confidence
prediction = model.predict(new_df_scaled)
confidence = model.predict_proba(new_df_scaled)

# output results
output_col1, output_col2 = st.columns([0.3, 0.7])
with output_col1.container(border=True):
    st.write("#### Your input:")
    for i in sample:
        st.write(sample[i].name, ":", sample[i][0])
    st.write("#####")
with output_col2.container(border=True):
    col2 = st.columns(2, gap="large")
    col2[0].write("### Prediction")
    with col2[0].container(border=True, width="content", vertical_alignment="center", horizontal_alignment="center"):
        st.header("High Rating" if prediction == 1 else "Low Rating")
        st.write("")
        
    col2[0].write(f"#### Confidence: {confidence[0][1] if prediction == 1 else confidence[0][0]:.4f} ({(confidence[0][1] if prediction == 1 else confidence[0][0])*100:.2f}%)")

    col2[0].text("Consistent daily logins and consuming materials represent genuine engagement which is heavily correlated with a High rating. While delete actions and forum actions often have lower importance scores because they occur less frequently or do not directly reflect primary learning progression metrics to the same extent as logging in, taking quizzes, or reading content. (P.S. No AI were used in making this project  <3)", text_alignment="justify")
    chart_data = importance_df.set_index('Feature')
    col2[1].write("### Model Feature Importance")
    col2[1].bar_chart(chart_data, horizontal=True, width="content")

    col2[1].write("#### Model Used in This Activity")
    col2[1].write(model)

# footer (about)
st.write("#")
margin1, margin3, footer1, margin4, margin2 = st.columns(5)
margin1.write("")
margin2.write("")
margin3.write("")
margin4.write("")
footer1.write("Developed by:")
footer1.write("### Rhoge Vhir A. Diaz")
footer1.write("##### BSCS 3")
footer1.write("##### CSDS 311")



