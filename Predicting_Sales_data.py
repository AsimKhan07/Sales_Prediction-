import streamlit as st
import joblib
import numpy as np
import pandas as pd
import datetime

# Load the trained model
model = joblib.load("gradient_boosting_model.pkl")

# ----- PAGE CONFIG -----
st.set_page_config(page_title="Sales Prediction App", layout="wide", page_icon="💼")

# ----- HEADER -----
st.title("💼 AI-Powered Sales Prediction App")
st.markdown("Get fast, accurate sales predictions using machine learning. \
             Predict based on individual orders or upload a batch CSV for bulk insights.")

# ----- TABS -----
tab1, tab2 = st.tabs(["📊 Single Prediction", "📂 Bulk Prediction (CSV)"])

# ----- SINGLE PREDICTION TAB -----
with tab1:
    st.header("🧮 Single Order Prediction")
    col1, col2 = st.columns(2)

    with col1:
        quantity = st.number_input("Quantity Ordered", min_value=1, step=1)
        price_each = st.number_input("Price Per Item", min_value=0.0, step=1.0)
        day = st.slider("Order Day", 1, 31, 15)

    with col2:
        weekday = st.selectbox(
            "Weekday",
            options=list(range(7)),
            format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
        )
        is_weekend = 1 if weekday >= 5 else 0
        st.markdown(f"**Weekend?** {'✅ Yes' if is_weekend else '❌ No'}")

    # Auto compute revenue per unit
    revenue_per_unit = quantity * price_each
    st.markdown(f"### 💰 Revenue per Unit: `{revenue_per_unit:.2f}`")

    features = np.array([quantity, price_each, day, weekday, is_weekend, revenue_per_unit]).reshape(1, -1)

    if st.button("🔮 Predict Sales"):
        prediction = model.predict(features)
        st.success(f"📈 Predicted Sales: **${prediction[0]:.2f}**")

# ----- BULK PREDICTION TAB -----
with tab2:
    st.header("📂 Upload CSV for Bulk Predictions")

    example_data = {
        "QUANTITYORDERED": [10, 20],
        "PRICEEACH": [45.0, 55.5],
        "DAY": [12, 23],
        "WEEKDAY": [1, 5],
    }
    st.markdown("**🧾 Sample CSV Structure:**")
    st.dataframe(pd.DataFrame(example_data))

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Compute extra fields
        df["IS_WEEKEND"] = df["WEEKDAY"].apply(lambda x: 1 if x >= 5 else 0)
        df["REVENUE_PER_UNIT"] = df["QUANTITYORDERED"] * df["PRICEEACH"]

        features = df[["QUANTITYORDERED", "PRICEEACH", "DAY", "WEEKDAY", "IS_WEEKEND", "REVENUE_PER_UNIT"]]
        predictions = model.predict(features)

        df["PREDICTED_SALES"] = predictions

        st.markdown("### ✅ Predictions")
        st.dataframe(df)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results as CSV", csv, "sales_predictions.csv", "text/csv")

# ----- FOOTER -----
st.markdown("---")
st.markdown("🔧 *Built with Streamlit & Scikit-learn | © 2025 Your Company Name*")
