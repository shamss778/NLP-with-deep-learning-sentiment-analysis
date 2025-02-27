import streamlit as st
import pickle
import time
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


# Initialize STOPWORDS
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text, return_tokens=False):
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', " ", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in STOPWORDS]
    
    # Remove extra white spaces (single-character tokens)
    tokens = [token for token in tokens if len(token) > 1]
    
    if return_tokens:
        return tokens
    else:
        return " ".join(tokens)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [preprocess_text(text) for text in X]

def to_dense(X):
    return X.toarray()

model = pickle.load(open('review_sentiment.pkl', 'rb'))
st.title("Review Sentiment Analysis")

review = st.text_input('Enter your review')

submit = st.button('Predict')

if submit:
    start = time.time()
    prediction = model.predict([review])
    end = time.time()
    st.write('Prediction time taken: ', round(end-start, 2), 'seconds')
    if prediction[0] == 0:
        p = 'Negative'
    elif prediction[0] == 1:
        p = 'Neutral'
    else:
        p = 'Positive'
    print(p)
    st.write(p)
