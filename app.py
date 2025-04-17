import streamlit as st
import pickle
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    

        nltk.download('wordnet')


with open('model.pkl','rb') as file:
    model = pickle.load(file)
    
with open('vector.pkl','rb') as file:
    vectorizer = pickle.load(file)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)


download_nltk_resources()
st.title("📩 Spam Email Classifier")
st.write("Enter a message below to check whether it's spam or not.")

user_input = st.text_area("✉️ Message")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        prob = model.predict_proba(vectorized_text)[0]

        if prediction == 1:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT SPAM**")

        st.write(f"Confidence: {round(max(prob)*100, 2)}%")

