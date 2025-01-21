import joblib
import numpy as np
import string
import streamlit as st

# Load the trained SVC model
svc_model = joblib.load('svc_model.pkl')

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase, remove punctuation, and split into words
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# Streamlit App Setup
st.markdown("""
    <style>
        /* Background styling */
        .main {
            background: linear-gradient(135deg, #ececec, #f5f5f5);
            color: #000; /* Dark color for text */
            padding: 20px;
            font-family: 'Arial', sans-serif;
        }
        /* Title styling */
        .title {
            color: #1a73e8; /* Blue shade for visibility */
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            text-shadow: 1px 1px #888888;
        }
        /* Subtitle styling */
        .subheader {
            font-size: 1.5em;
            font-weight: bold;
            color: #1a73e8; /* Blue shade */
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title for the app
st.markdown("<div class='title'>🔮 Sentiment Analysis of Movie Reviews 🔮</div>", unsafe_allow_html=True)

# Input: Movie review text
st.subheader("💡 Enter Movie Review")
review_text = st.text_area("Type the movie review here:")

# Function for sentiment prediction
def predict_sentiment(review):
    tokenized_review = preprocess_text(review)
    features = np.array([len(tokenized_review)])  # Simple example feature: length of the review
    features = features.reshape(1, -1)
    prediction = svc_model.predict(features)
    return "Positive" if prediction[0] == 0 else "Negative"

# Prediction on button click
if st.button("🔍 Predict Sentiment"):
    if review_text:
        sentiment = predict_sentiment(review_text)
        st.markdown(f"<h3 style='color: #1a73e8; text-align: center;'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
    else:
        st.error("Please enter a review to analyze.")
