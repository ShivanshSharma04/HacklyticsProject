from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
import nltk
import spacy
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")


training_data = [
    ("Blood pressure guidelines have been updated this year", "healthcare"),
    ("Yoga and meditation can alleviate stress", "healthcare"),
    ("The benefits of a plant-based diet", "healthcare"),
    ("Advancements in Alzheimer's research", "healthcare"),
    ("Understanding the side effects of chemotherapy", "healthcare"),
    ("How to perform CPR", "healthcare"),
    ("The impact of sleep on mental health", "healthcare"),
    ("Evaluating the efficacy of new diabetes treatments", "healthcare"),
    ("Preventive measures for heart disease", "healthcare"),
    ("Nutritional strategies for weight loss", "healthcare"),
    ("The premiere of the new superhero movie was last night", "not healthcare"),
    ("Tips for effective garden pest control", "not healthcare"),
    ("Exploring the art museums of Paris", "not healthcare"),
    ("The history of the Roman Empire", "not healthcare"),
    ("How to improve your photography skills", "not healthcare"),
    ("Beginner's guide to programming", "not healthcare"),
    ("The influence of social media on politics", "not healthcare"),
    ("Renewable energy sources and their potential", "not healthcare"),
    ("The economics of sustainable farming practices", "not healthcare"),
    ("Cultural impacts of globalization", "not healthcare"),
    ("Fever and coughing can be symptoms of the flu", "healthcare"),
    ("New study shows improvement in heart disease treatments", "healthcare"),
    ("The movie was thrilling and exciting", "not healthcare"),
    ("Delicious recipes for summer salads", "not healthcare"),
    ("Fever and coughing can be symptoms of the flu", "healthcare"),
    ("New study shows improvement in heart disease treatments", "healthcare"),
    ("The movie was thrilling and exciting", "not healthcare"),
    ("Delicious recipes for summer salads", "not healthcare"),
    ("Blood pressure guidelines have been updated this year", "healthcare"),
    ("Yoga and meditation can alleviate stress", "healthcare"),
    ("The benefits of a plant-based diet", "healthcare"),
    ("Advancements in Alzheimer's research", "healthcare"),
    ("Understanding the side effects of chemotherapy", "healthcare"),
    ("How to perform CPR", "healthcare"),
    ("The impact of sleep on mental health", "healthcare"),
    ("Evaluating the efficacy of new diabetes treatments", "healthcare"),
    ("Preventive measures for heart disease", "healthcare"),
    ("Nutritional strategies for weight loss", "healthcare"),
    ("The premiere of the new superhero movie was last night", "not healthcare"),
    ("Tips for effective garden pest control", "not healthcare"),
    ("Exploring the art museums of Paris", "not healthcare"),
    ("The history of the Roman Empire", "not healthcare"),
    ("How to improve your photography skills", "not healthcare"),
    ("Beginner's guide to programming", "not healthcare"),
]
# Preprocess the data and prepare for training
texts, labels = zip(*training_data)
text_train, text_test, label_train, label_test = train_test_split(texts, labels, test_size=0.1, random_state=26)

# Train a classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(text_train, label_train)

def extract_healthcare_entities(sentences):
    healthcare_entities = []
    for sentence in sentences:
        doc = nlp(sentence)
        # Extract entities tagged as 'DISEASE' or relevant types
        entities = [ent.text for ent in doc.ents if ent.label_ in ('DISEASE', 'ORGAN')]
        healthcare_entities.extend(entities)
    return healthcare_entities

# Function to predict and print healthcare-related strings
def classify_and_print_healthcare_strings(csv_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    df['Post Text'] = df['Post Text'].fillna('')

    # Tokenize 'Post Text' into individual sentences
    df['Sentences'] = df['Post Text'].apply(sent_tokenize)
    
    # Flatten the DataFrame to have one sentence per row
    sentences_df = df.explode('Sentences')
    sentences_df['Sentences'] = sentences_df['Sentences'].fillna('')

    # Predict categories for each sentence
    predictions = model.predict(sentences_df['Sentences'])
    sentences_df['Category'] = predictions
    
    # Filter and print healthcare-related sentences
    healthcare_sentences = sentences_df[sentences_df['Category'] == 'healthcare']['Sentences']
    print("Healthcare-related sentences:")
    for sentence in healthcare_sentences:
        print(sentence)


# Assuming the CSV file is named 'health.csv' and located in the same directory as the script
if __name__ == "__main__":
    classify_and_print_healthcare_strings('health.csv')
