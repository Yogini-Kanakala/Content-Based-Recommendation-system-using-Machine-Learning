import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

stemmer = PorterStemmer()
features = ['Product_ID', 'Name', 'Brand', 'Taxonomy_List', 'Keywords']
def preprocess_text(text):
    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    text = ' '.join(words)
    return text

def clean_taxonomy(raw_taxonomy):
    words = raw_taxonomy.split('|')
    words = [x for word in words for x in word.split('>')]
    words = [word.lower() for word in words]
    return "  ".join(words)


def preprocess_data(df):
    df.fillna('Missing', inplace=True)
    df.Name = df.Name.apply(preprocess_text)
    df.Taxonomy_List = df.Taxonomy_List.apply(clean_taxonomy)
    df.Brand = df.Brand.str.lower()
    df.Keywords = df.Keywords.apply(preprocess_text)
    return df
    

def generate_vectors(df):
    documents = (df.Name + " " + df.Brand + " " + df.Taxonomy_List + " " + df.Keywords).tolist()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_vectors = vectorizer.fit_transform(documents)
    nn_model = NearestNeighbors(n_neighbors=10, algorithm='brute')
    nn_model.fit(tfidf_vectors)
    return vectorizer, nn_model

def preprocess_input(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    text = ' '.join(words)
    return text
    
def recommend_products(query, vectorizer, nn_model, main_df, n=10):
    new_document = preprocess_input(query)
    new_tfidf_vector = vectorizer.transform([new_document])
    _, indices = nn_model.kneighbors(new_tfidf_vector,n_neighbors=n)
    res = main_df.iloc[indices[0]].reset_index(drop=True)
    return dict(res)
    