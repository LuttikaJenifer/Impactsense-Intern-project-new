import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import os
import joblib

model_path = "wine_model.joblib"

def train_and_save_model():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation Accuracy: {acc:.2f}")

    joblib.dump({
        "model": model,
        "feature_names": wine.feature_names,
        "target_names": wine.target_names
    }, model_path)

def load_model():
    if not os.path.exists(model_path):
        train_and_save_model()
    return joblib.load(model_path)

st.title("🍷 Wine Classification App")
st.write("This app predicts the type of wine (class 0, 1, or 2) based on chemical analysis.")

bundle = load_model()
model = bundle["model"]
feature_names = bundle["feature_names"]
target_names = bundle["target_names"]

st.sidebar.header("Input Features")
inputs = []

# Sliders for user input
for feat in feature_names[:5]:  
    val = st.sidebar.slider(
        feat,
        float(0),
        float(30),
        float(10)
    )
    inputs.append(val)

if st.sidebar.button("Predict"):
    arr = np.array(inputs).reshape(1, -1)

    df = pd.DataFrame([arr[0]], columns=feature_names[:5])
    for f in feature_names[5:]:
        df[f] = 0  # default 

    pred = model.predict(df)[0]
    probs = model.predict_proba(df)[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted Wine Class: {target_names[pred]}")
    st.write("Prediction probabilities:")
    st.bar_chart(pd.DataFrame(probs, index=target_names, columns=["Probability"]))
