import re
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Download NLTK resources
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# -----------------------------
# Load spaCy model
# -----------------------------
nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


def preprocess_sentence(sentence):

    sentence = sentence.lower()

    sentence = re.sub(r"\d+", "", sentence)

    sentence = re.sub(r"[^\w\s]", "", sentence)

    sentence = re.sub(r"\s+", " ", sentence).strip()

    doc = nlp(sentence)

    cleaned_words = []

    for token in doc:

        word = token.text

        if word not in stop_words:

            word = lemmatizer.lemmatize(word)

            cleaned_words.append(word)

    return " ".join(cleaned_words)