import joblib
import streamlit as st
import string
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained SVC model and vectorizer1
svc_model = joblib.load('svc_model.pkl')  # Load the trained model
vectorizer1 = joblib.load('vectorizer1.pkl')  # Load the TfidfVectorizer

# Function to preprocess text (tokenization and cleaning)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text  # Return processed text as a string (not a list of tokens)

# Function to predict sentiment based on the processed text
def predict_sentiment(review):
    # Step 1: Preprocess the review text
    processed_review = preprocess_text(review)
    
    # Step 2: Transform the processed review using the vectorizer1
    features = vectorizer1.transform([processed_review])  # Ensure the input is a list of strings
    
    # Step 3: Check the shape of the features (for debugging purposes)
    print(f"Features shape: {features.shape}")
    
    # Step 4: Make a prediction using the trained SVC model
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
