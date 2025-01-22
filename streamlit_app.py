import streamlit as st
import joblib
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.svm import SVC

# Ensure the models are available in the same directory as the script
vectorizer_path = os.path.join(os.getcwd(), 'vectorizer1.pkl')  # Path to the vectorizer model
svc_model_path = os.path.join(os.getcwd(), 'svc_model.pkl')  # Path to the SVC model

# Load the pre-trained models
try:
    vectorizer = joblib.load(vectorizer_path)
    svc_model = joblib.load(svc_model_path)
    st.write("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")

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

# Streamlit App with Customizations
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬", layout="wide")

# Title with custom color
st.markdown('<h1 style="color: #4CAF50;">Movie Review Sentiment Analysis 🎥</h1>', unsafe_allow_html=True)

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
        if sentiment == 1:
            st.success("Sentiment: Positive 🎬")
        else:
            st.error("Sentiment: Negative 🎭")
        
    else:
        st.error("Please enter a review.")
