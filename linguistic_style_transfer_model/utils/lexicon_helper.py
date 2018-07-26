from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords

from linguistic_style_transfer_model.config import global_config


def remove_words(tokens, words_to_remove):
    cleaned_tokens = list()
    for token in tokens:
        if token not in words_to_remove:
            cleaned_tokens.append(token)

    return cleaned_tokens


def get_sentiment_words():
    with open(file=global_config.sentiment_words_file_path,
              mode='r', encoding='ISO-8859-1') as sentiment_words_file:
        words = sentiment_words_file.readlines()
    words = set(word.strip() for word in words)

    return words


def get_stopwords():
    nltk_stopwords = set(stopwords.words('english'))
    sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS

    all_stopwords = set()
    all_stopwords = all_stopwords.union(spacy_stopwords)
    all_stopwords = all_stopwords.union(nltk_stopwords)
    all_stopwords = all_stopwords.union(sklearn_stopwords)

    return all_stopwords
