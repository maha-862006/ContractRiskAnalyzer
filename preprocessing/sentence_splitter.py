import re


def split_into_sentences(text):

    # Remove unnecessary spaces
    text = re.sub(r'\s+', ' ', text)

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Remove empty sentences
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences