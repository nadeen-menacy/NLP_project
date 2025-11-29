import streamlit as st
import joblib
import numpy as np
import os

# Force CPU to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="SMS Spam Detector", page_icon="📱")
st.title("📱 SMS Spam Detector")
st.write("Classify messages as **Spam** or **Ham** using classical and neural NLP models.")

# Choose model
model_options = ["NaiveBayes", "LogisticRegression", "LinearSVC", "BiLSTM"]
model_choice = st.selectbox("Choose a model", model_options)

# Input text
text = st.text_area("Enter your message:", height=120)

def predict_classical(model_name, text):
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    model = joblib.load(f"models/{model_name}.joblib")
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred

def predict_bilstm(text):
    tokenizer = joblib.load("models/lstm_tokenizer.joblib")
    model = load_model("models/bilstm_model.h5")
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=100, padding="post")
    pred = (model.predict(pad) > 0.5).astype("int32")[0][0]
    return "spam" if pred == 1 else "ham"

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a message.")
    else:
        if model_choice == "BiLSTM":
            result = predict_bilstm(text)
        else:
            result = predict_classical(model_choice, text)

        if result.lower() == "spam":
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **HAM (Not Spam)**")
