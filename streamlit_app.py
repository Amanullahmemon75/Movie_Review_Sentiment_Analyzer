import streamlit as st
import joblib
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC
import numpy as np


vectorizer = joblib.load(vectorizer_path)  # Load TF-IDF Vectorizer
svc_model = joblib.load(svc_model_path)  # Load the SVC model

# Initialize preprocessing tools
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize and remove stopwords
    return " ".join(words)  # Return the preprocessed text as a single string

# Streamlit App
st.title("Movie Review Sentiment Analysis")

review = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if review:
        # Preprocess the review text
        preprocessed_review = preprocess_text(review)
        
        # Transform the preprocessed review using the TF-IDF vectorizer
        review_tfidf = vectorizer.transform([preprocessed_review])  # Transform the input into TF-IDF features
        
        # Make sentiment prediction using the SVC model
        sentiment = svc_model.predict(review_tfidf)  # Predict using the SVC model
        
        # Display the sentiment result
        st.write(f"Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
        
    else:
        st.error("Please enter a review.")
