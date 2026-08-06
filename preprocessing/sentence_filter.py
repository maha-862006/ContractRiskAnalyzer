import re


def is_valid_sentence(sentence):
    """
    Returns True if the sentence should be analysed.
    Returns False if it is likely noise.
    """

    sentence = sentence.strip()

    # Empty sentence
    if not sentence:
        return False

    # Very short sentence
    if len(sentence.split()) < 5:
        return False

    # Only numbers
    if sentence.replace(".", "").isdigit():
        return False

    # Starts with Figure or Table
    if re.match(r"^(figure|table)\b", sentence, re.IGNORECASE):
        return False

    # Ignore common research-paper headings
    ignore_words = [
        "abstract",
        "keywords",
        "references",
        "acknowledgements",
        "appendix",
        "contents"
    ]

    if sentence.lower() in ignore_words:
        return False

    return True