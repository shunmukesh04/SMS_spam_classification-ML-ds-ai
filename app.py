import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

#lets load the saved vectorizer and naive model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

#tranform_text function for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
 
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
     
    
    text = nltk.word_tokenize(text)
    
    
    # Removing special characters and retaining alphanumeric words
    text = [word for word in text if word.isalnum()]
     
    stop_words = set(stopwords.words('english'))
    # Removing stopwords and punctuation
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
  
   
    text = [ps.stem(word) for word in text]
    
    
    return " ".join(text)

#saving strealit code
st.title("Email Spam Classifier")
input_sms = st.text_area("Enter message")

if st.button('Predict'):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])
     
    # Rest of the code remains the same...

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")