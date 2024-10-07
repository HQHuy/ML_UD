import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline

# Ensure nltk resources are available before running
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove digits
    text = re.sub(r'\d+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    # Filter out stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def perform_lemmatization(tokens):
    lemmatization = nltk.WordNetLemmatizer()
    # Perform lemmatization on tokens
    lemmatization_tokens = [lemmatization.lemmatize(token) for token in tokens]
    return lemmatization_tokens

def clean_text(text):
    # Preprocess text
    tokens = preprocess_text(text)
    # Remove stopwords
    filtered_tokens = remove_stopwords(tokens)
    # Perform lemmatization
    lemmatization_tokens = perform_lemmatization(filtered_tokens)
    # Return cleaned text as a string
    return ' '.join(lemmatization_tokens)


# Drug
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

def train_gaussian_nb(data):
    # Initialize the label encoders
    le_sex = LabelEncoder()
    le_bp = LabelEncoder()
    le_cholesterol = LabelEncoder()

    # Fit label encoders for categorical features
    data['Sex'] = le_sex.fit_transform(data['Sex'])
    data['BP'] = le_bp.fit_transform(data['BP'])
    data['Cholesterol'] = le_cholesterol.fit_transform(data['Cholesterol'])

    # Train GaussianNB model
    model = GaussianNB()
    model.fit(data.drop(columns=['Drug']), data['Drug'])  # Ensure 'Drug' is the target variable

    return model, le_sex, le_bp, le_cholesterol  # Return all three encoders

