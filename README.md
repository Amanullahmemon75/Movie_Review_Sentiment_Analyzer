# Movie_Review_Sentiment_Analyzer

## About the App
This app uses machine learning models to analyze movie reviews and determine their sentiment. Users can submit movie reviews in text form, and the app will predict whether the sentiment of the review is **Positive** or **Negative** based on a trained **Support Vector Classifier (SVC)** model and a **Random Forest Classifier (RFC)** model. The reviews are processed using **Word2Vec** embeddings to convert words into vector representations before prediction.

## Dataset

### Data Set
The labeled dataset consists of 25,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of the reviews is binary, meaning:
- IMDB rating < 5 results in a sentiment score of **"Negative"**.
- IMDB rating >= 7 results in a sentiment score of **"Positive"**.
- No individual movie has more than 30 reviews.

### File Description
- **MovieReviewTrainingDatabase.csv**: The labeled training set. The file is comma-delimited and contains a header row followed by 25,000 rows containing the sentiment and the text for each review.

### Data Fields
- **sentiment**: The sentiment of the review. "Positive" for positive reviews and "Negative" for negative reviews.
- **review**: The text of the review.

## Used Techniques
The app uses the following techniques for sentiment analysis:
- **Word2Vec**: A technique used for word embedding, converting words in the reviews into dense vectors to capture semantic meanings. The Word2Vec model helps to create vector representations of words in the reviews, which are then used for sentiment classification.
- **Support Vector Classifier (SVC)**: A machine learning classifier that helps predict sentiment based on the Word2Vec embeddings.
- **Random Forest Classifier (RFC)**: An additional classifier to predict the sentiment, providing a second opinion to the SVC model.

## How It Works
1. **User Input**: The user submits a movie review in text form.
2. **Preprocessing**: The app processes the text by converting it to lowercase, removing punctuation, and tokenizing it into individual words.
3. **Word Embedding**: Each tokenized word in the review is passed through a trained Word2Vec model to generate a vector representation of the words.
4. **Prediction**: The processed review vector is then passed to the trained models (SVC and RFC) to predict the sentiment:
   - **0 = Positive**
   - **1 = Negative**

The app then returns the sentiment prediction to the user as either **"Positive"** or **"Negative"**.

## Running the App
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

