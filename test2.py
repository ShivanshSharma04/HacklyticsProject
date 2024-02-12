import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
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
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai

openai.api_type = "azure"
openai.api_base = "https://aiatl-group-2.openai.azure.com/"
openai.api_version = "2023-09-15-preview"
openai.api_key = "211703b324ce43bf87805cceb4105cb2"

def query(prompt):
    # Define the initial system message
    system_message = {"role": "system", "content": "You are an AI assistant that helps people find information."}
    
    # Define the user message with the prompt provided to the function
    user_message = {"role": "user", "content": prompt}
    
    # Create a chat completion with the OpenAI API
    completion = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=[system_message, user_message],  # Include both the system and user messages
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    
    # Extract and return the response content from the API's response
    if completion.choices:
        response_content = completion.choices[0].message['content']
        return response_content
    else:
        return "No response received."

def visualize_buzzwords(text, visualization_type='wordcloud'):
    """
    Visualizes buzzwords from the provided text.

    :param text: The input text to analyze.
    :param visualization_type: The type of visualization ('wordcloud' or 'barchart').
    """
    # Tokenize and filter stopwords
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stop_words]
    
    # Generate a word frequency distribution
    freq_dist = FreqDist(words)
    
    if visualization_type == 'wordcloud':
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stop_words, min_font_size=10).generate_from_frequencies(freq_dist)
        # Plot the WordCloud image                        
        plt.figure(figsize=(8, 8), facecolor=None) 
        plt.imshow(wordcloud) 
        plt.axis("off") 
        plt.tight_layout(pad=0)
        plt.show()

    elif visualization_type == 'barchart':
        # Get most common words
        most_common_words = freq_dist.most_common(20)
        words = [word[0] for word in most_common_words]
        frequencies = [word[1] for word in most_common_words]
        
        # Plotting the bar chart
        plt.figure(figsize=(10, 8))
        plt.bar(words, frequencies)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.title('Top Buzzwords Frequency')
        plt.show()

    else:
        print("Invalid visualization type. Please choose 'wordcloud' or 'barchart'.")


def summarize_text(input_text):
    # Ensure NLTK resources are available
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    stopWords = set(stopwords.words("english"))
    words = word_tokenize(input_text)

    # Create a frequency table for word frequencies in the text
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Score sentences based on word frequencies
    sentences = sent_tokenize(input_text)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    # Calculate the average score for sentences
    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    average = int(sumValues / len(sentenceValue))

    # Select sentences for the summary based on their score
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (2 * average)):
            summary += " " + sentence

    return summary

def read_and_preprocess_texts_from_csv(file_path):
    """Read texts and comments from a CSV file, preprocess each, and return a list of preprocessed texts and comments."""
    df = pd.read_csv(file_path)
    preprocessed_texts = []
    preprocessed_comments = []
    
    for _, row in df.iterrows():
        # Preprocess the main text
        preprocessed_texts.append(row[df.columns[0]])
        
        # Preprocess each comment in the comments list
        preprocessed_comments.append(row[df.columns[1]])

    return preprocessed_texts, preprocessed_comments

def classify_text_with_bio_clinicalbert(text):
    """
    Classifies a given text as 'Health Related' or 'Not Health Related' using the Bio_ClinicalBERT model.
    
    Args:
    - text (str): The text to classify.
    
    Returns:
    - str: The predicted class.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Tokenize and encode the text for Bio_ClinicalBERT
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Move tensor to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Make predictions with the model
    with torch.no_grad():
        outputs = model(**inputs)

    # The model outputs logits
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get the predicted class (the one with the highest probability)
    predicted_class_index = predictions.argmax().item()

    # Map the predicted class index to the class name
    # Update this list based on the classes of your fine-tuned model
    classes = ['Not Health Related', 'Health Related']
    predicted_class = classes[predicted_class_index]

    return predicted_class

# Example usage
text = """Your text goes here. Make sure it's a long piece of text to better see the effect of the summarization algorithm."""
print(summarize_text(text))

file_path = '/Users/shivanshsharma/Desktop/Hacklytics/HacklyticsProject/health.csv'
all_data=read_and_preprocess_texts_from_csv(file_path)
#print(type(all_data))
#print(type(all_data[1]))

#print(all_data[1])
actual_all_data=""
for title in all_data[0]:
    actual_all_data+=str(title)

for i in range(len(all_data[1])):
    for j in all_data[1][i]:
       # print(type(j))
       # print(j)
        actual_all_data+=str(j)
        if(len(actual_all_data)>=70000):
            break

print(query("Give the top 5 health-related concerns that people have, given this list of popular post titles from subreddits relating to health. for each one of the top 5, you must give some quotes from the titles. giving quote is absolutely mandatory, and the most important part " + actual_all_data))
'''
print(len(actual_all_data))
print(len(all_data[0]))
print(len(all_data[1]))
quit()
#print(len(all_data))
#print(all_data[0])
'''
titles=""
for title in all_data[0]:
    titles+=str(title) + '.'
text = "Patient shows signs of chronic kidney disease with elevated creatinine levels."

for title in all_data[0]:
    if(classify_text_with_bio_clinicalbert(text)=='Health Related'):
        print("HEALTH RELATED:  ", title)
    else:
        print("NOT HEALTH RELATED ", title)


#for comment in all_data[1]:
#    print(summarize_text(comment))
#visualize_buzzwords(titles)

#print(summarize_text(titles))