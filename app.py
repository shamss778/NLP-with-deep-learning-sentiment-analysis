import streamlit as st
import pickle
import time
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