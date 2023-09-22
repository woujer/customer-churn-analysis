import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import numpy as np

# Load the pre-trained machine learning model
model = joblib.load("best_model.pkl")

# Load the original dataset
data = pd.read_csv("cleaned_data2.csv")
if st.checkbox("Show All Columns"):
    st.write(data)

st.title("Customer Churn Prediction")
st.title("Customer Churn Prediction")

# User input widgets
# Remove the age slider
# age = st.slider("Customer Age", min_value=float(data["Customer_Age"].min()), max_value=float(data["Customer_Age"].max()), value=float(data["Customer_Age"].median()))
credit_limit = st.slider("Credit Limit", min_value=float(data["Credit_Limit"].min()), max_value=float(data["Credit_Limit"].max()), value=float(data["Credit_Limit"].median()))
total_trans_ct = st.slider("Total Transaction Count", min_value=float(data["Total_Trans_Ct"].min()), max_value=float(data["Total_Trans_Ct"].max()), value=float(data["Total_Trans_Ct"].median()))

# Make predictions when the user clicks a button
if st.button("Predict"):
    # Ensure that the features match the format expected by the model
    # Remove the age feature
    features = [[credit_limit, total_trans_ct, int(data["Customer_Age"].median()), int(data["Credit_Limit"].median()), int(data["Total_Trans_Ct"].median()), 0, 0, 0, 0, 0, 0]]  # Use median for other features not specified
    prediction = model.predict(features)
    if prediction == 1:
        st.write("Prediction: Attrited Customer")
    else:
        st.write("Prediction: Existing Customer")

# Create age groups
age_bins = np.arange(data["Customer_Age"].min(), data["Customer_Age"].max() + 10, 10)
age_labels = [f"{int(b)}-{int(b)+9}" for b in age_bins[:-1]]
data["Age_Group"] = pd.cut(data["Customer_Age"], bins=age_bins, labels=age_labels, right=False)

selected_features = [
    'Total_Trans_Amt', 'Months_on_book',
    'Avg_Utilization_Ratio', 'Total_Revolving_Bal', 'Gender',
    'Income_Category_Less than $40K', 'Card_Category_Silver',
    'Income_Category_$80K - $120K', 'Card_Category_Gold', 'Credit_Limit'
]

# Create a new DataFrame to store age group, gender, and predictions
age_gender_prediction = data.groupby(["Age_Group", "Gender"]).size().reset_index(name="Count")
age_gender_prediction["Prediction"] = age_gender_prediction.apply(lambda row: model.predict_proba(row[selected_features].values.reshape(1, -1))[0][1], axis=1)

# Create an interactive bar plot to visualize the prediction by age group and gender
fig = px.bar(age_gender_prediction, x="Age_Group", y="Prediction", color="Gender", barmode="group", title="Prediction by Age Group and Gender")
st.plotly_chart(fig)
