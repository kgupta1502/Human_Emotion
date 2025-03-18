import re
import numpy as np
import pickle
import streamlit as st
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords if not already available
nltk.download('stopwords')

# Initialize stopwords set and stemmer
stopwords_set = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Load pre-trained TF-IDF vectorizer, logistic regression model, and label encoder
try:
    with open("logistic_regresion.pkl", "rb") as model_file:
        lg = pickle.load(model_file)
    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)
    with open("label_encoder.pkl", "rb") as label_file:
        lb = pickle.load(label_file)
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Make sure they are in the same directory.")

# Text cleaning function
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)  # Remove special characters
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]  # Apply stemming
    return " ".join(text)

# Emotion prediction function
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    
    # Apply the vectorizer
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]

    # Get probability scores
    probabilities = lg.predict_proba(input_vectorized)[0]
    confidence_score = np.max(probabilities)

    return predicted_emotion, confidence_score

# Streamlit App UI
st.title("Six Human Emotions Detection App")
st.write("=================================================")
st.write("['Joy', 'Fear', 'Anger', 'Love', 'Sadness', 'Surprise']")
st.write("=================================================")

# Take user input
user_input = st.text_input("Enter your text here:")

if st.button("Predict"):
    if user_input.strip():  # Check if input is not empty
        predicted_emotion, confidence = predict_emotion(user_input)
        st.write("Predicted Emotion:", predicted_emotion)
        st.write(f"Confidence Score: {confidence:.2f}")
    else:
        st.warning("Please enter some text for prediction.")
