import joblib
import numpy as np
import string
import streamlit as st

# Load the trained SVC model and vectorizer
svc_model = joblib.load('svc_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # Load the vectorizer used during training

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase, remove punctuation, and split into words
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Streamlit App Setup
st.markdown("""
    <style>
        .main {
            background: #000;  /* Black background */
            color: #fff;  /* White text */
            padding: 20px;
            font-family: 'Arial', sans-serif;
        }
        .title {
            color: #FFD700;  /* Gold for movie theme */
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            text-shadow: 2px 2px #333;
        }
        .subheader {
            font-size: 1.5em;
            font-weight: bold;
            color: #FFD700;  /* Gold for subheader */
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title for the app
st.markdown("<div class='title'>🎬 Sentiment Analysis of Movie Reviews 🎥</div>", unsafe_allow_html=True)

# Input: Movie review text
st.subheader("📝 Enter Your Movie Review")
review_text = st.text_area("Type the movie review here:")

# Function for sentiment prediction
def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)
    features = vectorizer.transform([preprocessed_review])  # Convert input to feature vector
    prediction = svc_model.predict(features)
    return "Positive 🎉" if prediction[0] == 1 else "Negative 😔"

# Prediction on button click
if st.button("🎬 Predict Sentiment"):
    if review_text.strip():
        sentiment = predict_sentiment(review_text)
        st.markdown(f"<h3 style='color: #FFD700; text-align: center;'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
    else:
        st.error("Please enter a review to analyze.")
