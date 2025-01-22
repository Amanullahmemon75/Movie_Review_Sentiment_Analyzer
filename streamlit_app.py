import joblib
import numpy as np
import string
from gensim.models import Word2Vec
import streamlit as st
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load the trained models
svc_model = joblib.load('svc_model.pkl')  # Load the trained SVC model
vectorizer = joblib.load('vectorizer1.pkl')  # Load the TfidfVectorizer model
word2vec_model = Word2Vec.load('word2vec_model')  # Load the trained Word2Vec model

# Function to preprocess text (tokenization and cleaning)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text  # Return processed text as a string (not a list of tokens)

# Function to calculate Average Word2Vec for a sentence
def avg_word2vec(sentence, model):
    vec = np.zeros(200)  # Size of the word vectors (should match the Word2Vec vector size)
    count = 0
    for word in sentence:
        if word in model.wv:
            vec += model.wv[word]
            count += 1
    if count > 0:
        vec /= count
    return vec

# Function to predict sentiment based on the processed text
def predict_sentiment(review):
    # Step 1: Preprocess the review text
    processed_review = preprocess_text(review)
    
    # Step 2: Tokenize the preprocessed review text using the loaded vectorizer
    tokenized_review = vectorizer.transform([processed_review])  # Transform the review into a feature vector
    
    # Step 3: Calculate the Average Word2Vec for the review
    avg_vec = avg_word2vec(processed_review.split(), word2vec_model)
    
    # Step 4: Combine the features (Tfidf + Word2Vec features)
    # You can concatenate the TF-IDF and Word2Vec features if necessary, 
    # or just use one of them based on your model design.
    
    # For now, we only use the Word2Vec-based features
    features = np.array([avg_vec])  # This is a NumPy array with the average Word2Vec vector
    
    # Step 5: Make a prediction using the trained SVC model
    prediction = svc_model.predict(features)
    
    # Return the sentiment based on the prediction (assuming binary: 1 = Positive, 0 = Negative)
    return "Positive 🎉" if prediction[0] == 1 else "Negative 😔"

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

# Prediction on button click
if st.button("🎬 Predict Sentiment"):
    if review_text.strip():  # Ensure there is some text to process
        sentiment = predict_sentiment(review_text)
        st.markdown(f"<h3 style='color: #FFD700; text-align: center;'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
    else:
        st.error("Please enter a review to analyze.")
