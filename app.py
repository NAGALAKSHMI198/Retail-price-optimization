import streamlit as st
import pandas as pd
from xgboost import XGBRegressor   # ✅ IMPORTANT

st.title("Retail Price Optimization")

# Load data
df = pd.read_csv("reatail.csv")

# Show data
st.write("Dataset Preview")
st.dataframe(df.head())

# Features
X = df[['price','competitor_price','discount','cogs','month']]
y = df['sales']

# Train model
model = XGBRegressor()
model.fit(X, y)

# Inputs
st.subheader("Enter Product Details")

price = st.slider("Price", 50, 500)
competitor = st.slider("Competitor Price", 50, 500)
discount = st.slider("Discount", 0, 30)
cogs = st.slider("COGS", 30, 300)
month = st.slider("Month", 1, 12)

# Prediction
if st.button("Predict"):
    demand = model.predict([[price, competitor, discount, cogs, month]])
    profit = (price - cogs) * demand[0]

    st.success(f"Predicted Demand: {demand[0]:.2f}")
    st.success(f"Estimated Profit: {profit:.2f}")