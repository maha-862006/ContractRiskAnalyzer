import re
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


def preprocess_sentence(sentence):

    # Lowercase
    sentence = sentence.lower()

    # Remove numbers
    sentence = re.sub(r"\d+", "", sentence)

    # Remove punctuation
    sentence = re.sub(r"[^\w\s]", "", sentence)

    # Remove extra spaces
    sentence = re.sub(r"\s+", " ", sentence).strip()

    # spaCy tokenization
    doc = nlp(sentence)

    cleaned_words = []

    for token in doc:

        word = token.text

        if word not in stop_words:

            word = lemmatizer.lemmatize(word)

            cleaned_words.append(word)

    return " ".join(cleaned_words)