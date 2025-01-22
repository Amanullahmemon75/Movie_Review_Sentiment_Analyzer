import streamlit as st
import joblib
import os
import string
import nltk
import requests
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.svm import SVC
import numpy as np

# Define paths to store the downloaded models
models_dir = 'models'  # Directory where models will be saved
vectorizer1_filename = 'vectorizer1.pkl'
svc_model_filename = 'svc_model.pkl'

# Ensure the models directory exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# URLs of the models in your GitHub repository (replace these with actual URLs)
vectorizer1_url = "https://github.com/Amanullahmemon75/Movie_Review_Sentiment_Analyzer/tree/main/vectorizer1.pkl"
svc_model_url = "https://github.com/Amanullahmemon75/Movie_Review_Sentiment_Analyzer/tree/main/svc_model.pkl"

# Download model files from GitHub
def download_model(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(models_dir, filename), 'wb') as f:
            f.write(response.content)
    else:
        st.error(f"Failed to download {filename}")

# Download both models
download_model(vectorizer1_url, vectorizer1_filename)
download_model(svc_model_url, svc_model_filename)

# Load the pre-trained models
vectorizer_path = os.path.join(models_dir, vectorizer1_filename)
svc_model_path = os.path.join(models_dir, svc_model_filename)

# Check if the models are successfully downloaded before loading
if os.path.exists(vectorizer_path) and os.path.exists(svc_model_path):
    vectorizer = joblib.load(vectorizer_path)
    svc_model = joblib.load(svc_model_path)
else:
    st.error("Failed to load models. Please check the file paths and ensure they are downloaded correctly.")

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
