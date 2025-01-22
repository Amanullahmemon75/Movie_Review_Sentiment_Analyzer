import streamlit as st
import joblib
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from gensim.models import Word2Vec
from sklearn.svm import SVC

# Load the pre-trained models
vectorizer_path = os.path.join('models', 'vectorizer1.pkl')  # For TF-IDF Vectorizer
svc_model_path = os.path.join('models', 'svc_model.pkl')  # Trained SVC model

vectorizer = joblib.load(vectorizer_path)
svc_model = joblib.load(svc_model_path)

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
    return words  # Return preprocessed text as list of words

# Function to calculate Average Word2Vec for a sentence
def avg_word2vec(sentence, model):
    vec = np.zeros(200)  # Size of the word vectors (200-dimensional embeddings)
    count = 0
    for word in sentence:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count > 0:
        vec /= count
    return vec

# Streamlit App
st.title("Movie Review Sentiment Analysis")

review = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if review:
        # Preprocess the review
        preprocessed_review = preprocess_text(review)
        
        # Calculate average Word2Vec for the preprocessed review
        avg_vector = avg_word2vec(preprocessed_review, word2vec_model)
        
        # Make sentiment prediction using the SVC model
        sentiment = svc_model.predict([avg_vector])  # Use the average Word2Vec vector for prediction
        
        # Display the sentiment result
        st.write(f"Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
        
    else:
        st.error("Please enter a review.")
