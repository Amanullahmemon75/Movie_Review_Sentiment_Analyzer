import joblib
import streamlit as st
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained vectorizer1 model
vectorizer1 = joblib.load('vectorizer1.pkl')  # Load the TfidfVectorizer

# Function to preprocess text (tokenization and cleaning)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text  # Return processed text as a string (not a list of tokens)

# Function to transform review into vectorized features using the vectorizer1
def transform_review(review):
    # Step 1: Preprocess the review text
    processed_review = preprocess_text(review)
    
    # Step 2: Transform the processed review using vectorizer1
    features = vectorizer1.transform([processed_review])  # Ensure the input is a list of strings
    
    return features

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
st.markdown("<div class='title'>🎬 Transform Your Movie Reviews 🎥</div>", unsafe_allow_html=True)

# Input: Movie review text
st.subheader("📝 Enter Your Movie Review")
review_text = st.text_area("Type the movie review here:")

# Transformation on button click
if st.button("🎬 Transform Review"):
    if review_text.strip():  # Ensure there is some text to process
        transformed_features = transform_review(review_text)
        st.markdown(f"<h3 style='color: #FFD700; text-align: center;'>Transformed Features:</h3>", unsafe_allow_html=True)
        st.write(transformed_features.toarray())  # Display the transformed features (vectorized representation)
    else:
        st.error("Please enter a review to transform.")
