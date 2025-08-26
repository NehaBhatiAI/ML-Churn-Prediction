import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ===============================
# Load model and scaler
# ===============================
model = pickle.load(open("NB_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

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
                autopct='%1.1f%%', labels=['Not Churned','Churned'], 
                colors=['#1f77b4','#ff7f0e'], ax=ax
            )
            ax.set_ylabel("")
            st.pyplot(fig)

        with col5:
            st.subheader("Churn by Age Group")
            df['AgeGroup'] = pd.cut(df['Age'], bins=[18,30,40,50,60,100], 
                                    labels=["18-30","31-40","41-50","51-60","60+"])
            churn_by_age = df.groupby('AgeGroup')['Exited'].mean()*100
            st.bar_chart(churn_by_age)

    else:
        st.warning("Please upload data first.")

# ===============================
# EDA
# ===============================
# EDA
elif menu == "EDA & Visualization":
    st.title("📈 Exploratory Data Analysis")
    if "df" in st.session_state:
        df = st.session_state["df"]

        # Select only numeric columns for correlation
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

    # Inputs (you can adjust these as per your trained model)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    balance = st.number_input("Balance", step=100.0)
    num_products = st.slider("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
    is_active_member = st.selectbox("Is Active Member?", [0, 1])

    if st.button("Predict"):
        input_data = [[credit_score, age, balance, num_products, has_cr_card, is_active_member]]

        # Scale input before prediction
        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)[0]
        st.success("🔴 Churn" if prediction==1 else "🟢 Not Churn")
