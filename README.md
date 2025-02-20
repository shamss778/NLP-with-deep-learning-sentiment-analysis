# NLP-with-deep-learning-sentiment-analysis

## Project Overview

This project analyzes customer reviews for Dabchy, a Tunisian online clothing marketplace application. The goal is to understand customer sentiment and provide actionable insights to optimize user experience and improve customer loyalty. We follow the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology. We have also integrated a Streamlit-based API. It allows users to input their review and predict if it's positive, neutral or negative.

### Data Source

*   Google Play Store reviews were collected using the SerpAPI.

### Data Extraction

The following data fields were extracted:

*   `title` (Review Title)
*   `rating` (Star Rating)
*   `snippet` (Review Text)
*   `Likes` (Number of Likes)
*   `Date` (Review Date)

The extracted data is stored in the `app_reviews.csv` file.

### Initial Data Assessment

*   The initial dataset was imbalanced.
*   There were redundant data entries.
*   There was 2 empty comments.

### Data Cleaning
*   Redundant data and empty comments were removed.

### Sentiment Classification

*   Reviews were classified into three categories based on their star rating:
    *   **Positive:** rating = 5
    *   **Neutral:** 1 < rating < 5
    *   **Negative:** score = 1

### Text Preprocessing Steps:

1.  **Translation:** Translated all reviews to English.
2.  **Lowercasing:** Converted all text to lowercase.
3.  **Punctuation Removal:** Removed punctuation (except exclamation marks, question marks, colons, semicolons, and quotes).
4.  **Stop Word Removal:** Removed common stop words.
5.  **Lemmatization:** Applied lemmatization to reduce words to their base form.

### Data Augmentation (EDA)

To address data imbalance and improve model robustness, data augmentation techniques were employed based on this paper: [https://arxiv.org/abs/1901.11196](https://arxiv.org/abs/1901.11196)

*   BERT-based Data Augmentation: Utilizes contextual word embeddings from BERT to intelligently insert new words into existing reviews, enhancing dataset diversity while preserving semantic meaning.
*   Customizable Augmentation Process: Implements a flexible function augmentMyData() that allows for targeted augmentation of minority classes, with adjustable parameters for repetition count and sample size to fine-tune the augmentation strategy.

### Post-Augmentation Data Assessment

*   No redundant data entries.
*   No empty values.
*   Most reviews were relatively short.

### Model Used

Supervised machine learning models were used for sentiment classification:

*   Gaussian Naive Bayes

## Running the Streamlit App

1. Install Streamlit: pip install streamlit
2. Launch the Streamlit app: streamlit run app.py


