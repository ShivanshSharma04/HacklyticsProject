# Imports and Setup
import nltk
from nltk.corpus import stopwords, names
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import ast
from nltk.corpus import wordnet
from nltk import pos_tag
# Additional imports...
from collections import Counter

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('names')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
# More nltk.download calls...

def get_wordnet_pos(treebank_tag):
    """Convert treebank POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None  # If no match, return None

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Convert text to string to avoid AttributeError on float objects
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)

    # POS tagging
    words_pos = pos_tag(words)

    # Lemmatization with POS tags
    lemmatized_words = []
    for word, pos in words_pos:
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN  # Default to NOUN if None
        if word.isalnum() and word not in stop_words:
            lemmatized_word = lemmatizer.lemmatize(word, wordnet_pos)
            lemmatized_words.append(lemmatized_word)

    return lemmatized_words

def read_and_preprocess_texts_from_csv(file_path):
    """Read texts and comments from a CSV file, preprocess each, and return a list of preprocessed texts and comments."""
    df = pd.read_csv(file_path)
    preprocessed_texts = []
    preprocessed_comments = []
    
    for _, row in df.iterrows():
        # Preprocess the main text
        preprocessed_texts.append(preprocess_text(row[df.columns[0]]))
        
        # Preprocess each comment in the comments list
        try:
            # Attempt to convert the second column to a list (assuming it's stored as a string representation of a list)
            comments = ast.literal_eval(row[df.columns[1]])
            if isinstance(comments, list):
                preprocessed_comment_list = [preprocess_text(comment) for comment in comments]
                preprocessed_comments.append(preprocessed_comment_list)
            else:
                preprocessed_comments.append([])
        except (ValueError, SyntaxError):
            # If conversion fails or the column does not contain a properly formatted list, append an empty list for comments
            preprocessed_comments.append([])
    
    return preprocessed_texts, preprocessed_comments

def extract_significant_terms(preprocessed_texts):
    # Implementation...
    """Extract significant terms based on frequency from already preprocessed texts."""
    all_words = []
    for words_list in preprocessed_texts:
        all_words.extend(words_list)  # Here, words_list is already a list of lemmatized words
    
    # Calculate word frequency
    word_freq = Counter(all_words)
    
    # Filter out non-informative words
    non_informative_words = {'health', 'disease', 'condition'}
    significant_words = {word: freq for word, freq in word_freq.items() if word not in non_informative_words}
    
    # Sort words by frequency
    sorted_words = sorted(significant_words.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_words[:1000]

def perform_sentiment_analysis(texts):
    # Implementation...
    """Perform sentiment analysis on a list of texts, returning the overall sentiment."""
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [sia.polarity_scores(text) for text in texts]
    
    return sentiment_scores

def topics_with_sentiment(preprocessed_texts):
    """Identify topics with their sentiment scores."""
    sia = SentimentIntensityAnalyzer()
    
    # Flatten the list of preprocessed_texts to perform frequency analysis
    all_words_flat = [word for sublist in preprocessed_texts for word in sublist]
    word_freq = Counter(all_words_flat)
    
    # Identify the most common terms, excluding non-informative words
    non_informative_words = {'health', 'disease', 'condition'}
    significant_terms = [(word, freq) for word, freq in word_freq.items() if word not in non_informative_words][:1000]
    
    # For each significant term, find texts that contain this term and perform sentiment analysis
    term_sentiments = {}
    for term, _ in significant_terms:
        term_texts = [" ".join(text) for text in preprocessed_texts if term in text]
        if term_texts:
            # Average sentiment scores for texts containing the term
            term_sentiment_scores = [sia.polarity_scores(text)['compound'] for text in term_texts]
            avg_sentiment = sum(term_sentiment_scores) / len(term_sentiment_scores)
            term_sentiments[term] = avg_sentiment
        else:
            term_sentiments[term] = 0  # Default sentiment score if term not found in any text
    
    # Pair each term with its average sentiment score
    term_sentiment_pairs = [(term, term_sentiments[term]) for term in term_sentiments]
    
    # Sort terms by their sentiment scores for demonstration
    term_sentiment_pairs_sorted = sorted(term_sentiment_pairs, key=lambda x: x[1], reverse=True)
    
    return term_sentiment_pairs_sorted


# Main Logic
def main():
    '''
    # Example main logic that uses the functions defined above
    texts = ["Sample text from subreddit posts and comments..."]
    processed_text = preprocess_text(texts)
    topics_with_sentiment = topics_with_sentiment(processed_text)
    # Process and print results...
    print(topics_with_sentiment)
    '''
    # Adjust the file path as necessary
    file_path = '/Users/shivanshsharma/Desktop/Hacklytics/health.csv'
    preprocessed_texts, _ = read_and_preprocess_texts_from_csv(file_path)
    
    # Extract significant terms and their sentiments
    term_sentiment_pairs = topics_with_sentiment(preprocessed_texts)
    
    print(term_sentiment_pairs)
if __name__ == "__main__":
    main()
