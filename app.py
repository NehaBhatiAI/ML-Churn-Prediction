import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ===============================
# Load model, scaler, features
# ===============================
model = pickle.load(open("NB_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="Bank Churn Dashboard", layout="wide")

# ===============================
# Sidebar navigation
# ===============================
menu = st.sidebar.radio(
    "Navigation",
    ["Upload Data", "Dashboard", "EDA & Visualization", "Prediction"]
)

# ===============================
# Upload Data
# ===============================
if menu == "Upload Data":
    st.title("📂 Upload Customer Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("✅ Data uploaded successfully!")
    else:
        st.info("Upload a CSV file to proceed.")

# ===============================
# Dashboard
# ===============================
elif menu == "Dashboard":
    st.title("📊 Customer Churn Dashboard")
    if "df" in st.session_state:
        df = st.session_state["df"]

        # Metrics
        col1, col2, col3 = st.columns(3)
        churn_rate = df['Exited'].mean()*100
        col1.metric("Churn Rate", f"{churn_rate:.2f}%")
        col2.metric("Total Customers", len(df))
        col3.metric("Active Customers", len(df[df['Exited']==0]))

        # Charts
        col4, col5 = st.columns(2)
        with col4:
            st.subheader("Churn Distribution")
            fig, ax = plt.subplots()
            df['Exited'].value_counts().plot.pie(
                autopct='%1.1f%%',
                labels=['Not Churned','Churned'],
                colors=['#1f77b4','#ff7f0e'],
                ax=ax
            )
            ax.set_ylabel("")
            st.pyplot(fig)

        with col5:
            st.subheader("Churn by Age Group")
            df['AgeGroup'] = pd.cut(
                df['Age'],
                bins=[18,30,40,50,60,100],
                labels=["18-30","31-40","41-50","51-60","60+"]
            )
            churn_by_age = df.groupby('AgeGroup')['Exited'].mean()*100
            st.bar_chart(churn_by_age)

    else:
        st.warning("Please upload data first.")

# ===============================
# EDA
# ===============================
elif menu == "EDA & Visualization":
    st.title("📈 Exploratory Data Analysis")
    if "df" in st.session_state:
        df = st.session_state["df"]

        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available for correlation heatmap.")
    else:
        st.warning("Upload data first.")

# ===============================
# Prediction
# ===============================
elif menu == "Prediction":
    st.title("🤖 Predict Customer Churn")
    st.subheader("Single Customer Prediction")

    # Collect user inputs
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, step=1)
    balance = st.number_input("Balance", step=100.0)
    num_products = st.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
    is_active_member = st.selectbox("Is Active Member?", [0, 1])
    est_salary = st.number_input("Estimated Salary", step=100.0)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])  # adjust to dataset

    if st.button("Predict"):
        # Build input dataframe
        input_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Gender": 1 if gender=="Male" else 0,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": est_salary,
            "Geography": geography
        }])

        # One-hot encode categorical variables
        input_df = pd.get_dummies(input_df, drop_first=True)

        # Reindex to training features
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        st.success("🔴 Churn" if prediction==1 else "🟢 Not Churn")
