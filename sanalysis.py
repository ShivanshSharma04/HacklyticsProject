import pandas as pd
from textblob import TextBlob
import ast  # To convert string representation of list to list
import disease_names as dn  # Assuming this module provides a list of disease names
from collections import Counter
from difflib import SequenceMatcher

# No need to download NLTK resources for basic tokenization in this approach

# Custom function to preprocess and tokenize text
def preprocess_and_tokenize(text, healthcare_terms_set):
    # Convert to lowercase
    text = text.lower()
    # Tokenize into phrases based on the healthcare terms
    tokens = [term for term in healthcare_terms_set if term in text]
    return tokens

# Calculate sentiment
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def main():
    # Load the dataset
    df = pd.read_csv('/Users/shivanshsharma/Desktop/Hacklytics/HacklyticsProject/health.csv', converters={'second_column_name': ast.literal_eval})

    # Ensure the first column is treated as a string
    df['Post Text'] = df['Post Text'].astype(str)

    # Concatenate the first column with the list of strings from the second column
    df['combined_text'] = df.apply(lambda row: row['Post Text'] + ' ' + ' '.join(row['Comments']), axis=1)

    # Get a predefined list of healthcare-related terms
    healthcare_terms = dn.get_disease_names()  # This should be a list of terms
    print(healthcare_terms)
    #healthcare_terms = ['cancer','health','food','god']

    # Convert the list of terms into a set for faster lookup
    healthcare_terms_set = set(healthcare_terms)

    # Process text and filter healthcare-related terms
    df['healthcare_related'] = df['combined_text'].apply(lambda x: preprocess_and_tokenize(x, healthcare_terms_set))

    # Frequency analysis
    all_healthcare_terms = sum(df['healthcare_related'], [])
    healthcare_terms_freq = Counter(all_healthcare_terms)
    print("Frequency of Healthcare-Related Terms:", healthcare_terms_freq)

    # Sentiment analysis
    df['sentiment'] = df['combined_text'].apply(calculate_sentiment)
    print("Average Sentiment:", df['sentiment'])

if __name__ == "__main__":
    main()
