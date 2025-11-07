# Placeholder Streamlit app (to be completed later)
import streamlit as st
import pandas as pd

st.title("NBA MVP Predictor — Dashboard (Prototype)")
st.write("Chargez vos features et vos prédictions pour explorer les résultats.")
uploaded = st.file_uploader("Upload predictions CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(20))
