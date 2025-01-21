import joblib
import numpy as np
from flask import Flask, request, jsonify
import string

# Load the trained SVC model and Random Forest model
svc_model = joblib.load('svc_model.pkl')

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase, remove punctuation, and split into words
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review = data.get('review')  # Input review

        if not review:
            return jsonify({'error': 'Review text is required'}), 400

        tokenized_review = preprocess_text(review)  # Preprocess review

        # Here, you may need to create a different feature vector for SVC or Random Forest model
        # Assuming you have a method to transform the tokenized text into numerical features for the models
        features = np.array([len(tokenized_review)])  # Example feature: length of the review, update with actual features

        features = features.reshape(1, -1)
        prediction = svc_model.predict(features)
        
        # Correct sentiment mapping: 0 = Positive, 1 = Negative
        sentiment = 'Positive' if prediction[0] == 0 else 'Negative'
        
        return jsonify({'review': review, 'sentiment': sentiment})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
