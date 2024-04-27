from flask import Flask, request, render_template, jsonify
import mysql.connector
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # Import Multinomial Naive Bayes
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Connect to the MySQL database
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='12345',
    database='jobofferlist'
)
cursor = conn.cursor()

# Fetch data from the database
cursor.execute("SELECT job_title, skills_categories, skills FROM jobofferlist")
data = cursor.fetchall()

# Close the database connection
conn.close()

# Convert fetched data to a DataFrame
df = pd.DataFrame(data, columns=['job_title', 'skills_categories', 'skills'])

# Step 2: Data Preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

df['job_title'] = df['job_title'].apply(preprocess_text)
df['skills_categories'] = df['skills_categories'].apply(preprocess_text)

# Step 3: Define a function to train the model
def train_model():
    # Split Data
    X = df[['job_title', 'skills_categories']]
    y = df['skills']  # We won't use y in MultinomialNB since it's unsupervised
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Use TF-IDF to vectorize text data
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train['job_title'] + ' ' + X_train['skills_categories'])

    # Train Multinomial Naive Bayes Classifier
    nb_model = MultinomialNB()  
    nb_model.fit(X_train_tfidf, X_train.index)  

    # Calculate and print model accuracy
    accuracy = nb_model.score(X_train_tfidf, X_train.index)

    # Save the trained model and vectorizer
    joblib.dump(nb_model, 'models/nb_model.joblib')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')

    return accuracy

# Step 5: Define route and function for Flask app (similar to before)
def retrieve_similar_jobs(job_title, skills_categories):
    try:
        # Load the saved model and vectorizer
        vectorizer = joblib.load('models/vectorizer.joblib')
        nb_model = joblib.load('models/nb_model.joblib')

        # Preprocess input text
        job_title = preprocess_text(job_title)
        skills_categories = preprocess_text(skills_categories)
        
        # Vectorize input text
        input_text_tfidf = vectorizer.transform([job_title + ' ' + skills_categories])
        
        # Use Naive Bayes Classifier to find similar jobs
        similar_job_indices = nb_model.predict(input_text_tfidf)
        similar_job_titles = [df.iloc[i]['job_title'] for i in similar_job_indices.flatten()]  
        relevant_skills = [df.iloc[i]['skills'] for i in similar_job_indices.flatten()] 
        
        return similar_job_titles, relevant_skills
    except Exception as e:
        print("Error:", e)
        return ["Error occurred while retrieving similar jobs"], [None]
