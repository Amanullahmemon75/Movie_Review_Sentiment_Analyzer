import streamlit as st
import joblib
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Set the page config at the very beginning
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬", layout="wide")

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Set paths for models (make sure these paths are correct in your environment)
vectorizer_path = 'vectorizer1.pkl'  # Path to the TF-IDF vectorizer model
svc_model_path = 'svc_model.pkl'     # Path to the SVC model

# Load the pre-trained models
try:
    vectorizer = joblib.load(vectorizer_path)
    svc_model = joblib.load(svc_model_path)
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess function
def preprocess_text(text):
    """
    Preprocess the input text by converting it to lowercase,
    removing punctuation, and lemmatizing words.
    """
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return " ".join(words)  # Return the preprocessed text as a single string

# Function to analyze sentiment
def analyze_sentiment(review):
    """
    Analyzes the sentiment of a given review text.
    Returns 'Positive' or 'Negative' based on the SVC model prediction.
    """
    # Preprocess the review text
    preprocessed_review = preprocess_text(review)
    
    # Transform the preprocessed review using the TF-IDF vectorizer
    review_tfidf = vectorizer.transform([preprocessed_review])  # Transform the input into TF-IDF features
    
    # Make sentiment prediction using the SVC model
    sentiment = svc_model.predict(review_tfidf)  # Predict using the SVC model
    
    # Return the sentiment result
    return "Positive" if sentiment == 1 else "Negative"

# Streamlit App
st.title("🎬 Movie Review Sentiment Analysis")

review = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if review:
        # Analyze sentiment
        sentiment_result = analyze_sentiment(review)
        
        # Display the sentiment result
        st.write(f"Sentiment: {sentiment_result}")
        
    else:
        st.error("Please enter a review.")
