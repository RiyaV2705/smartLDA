"""Text preprocessing pipeline for topic modeling."""

import re
from typing import List, Tuple, Set, Optional
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Attempt NLTK lemmatization / stopwords with safe fallbacks
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords as nltk_stopwords

    try:
        nltk.data.find("corpora/stopwords")
    except (LookupError, AttributeError):
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except (LookupError, AttributeError):
        nltk.download("wordnet", quiet=True)

    try:
        NLTK_STOPWORDS = set(nltk_stopwords.words("english"))
    except Exception:
        NLTK_STOPWORDS = set(ENGLISH_STOP_WORDS)

    try:
        LEMMATIZER = WordNetLemmatizer()
    except Exception:
        LEMMATIZER = None
except Exception:
    NLTK_STOPWORDS = set(ENGLISH_STOP_WORDS)
    LEMMATIZER = None


DEFAULT_STOPWORDS: Set[str] = set(ENGLISH_STOP_WORDS).union(NLTK_STOPWORDS)


class TextPreprocessor:
    """Preprocesses raw documents and transforms them into a Document-Term Matrix."""

    def __init__(
        self,
        remove_numbers: bool = True,
        apply_lemmatization: bool = True,
        min_word_len: int = 3,
        custom_stopwords: Optional[List[str]] = None,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        max_features: Optional[int] = 5000,
    ):
        self.remove_numbers = remove_numbers
        self.apply_lemmatization = apply_lemmatization and (LEMMATIZER is not None)
        self.min_word_len = min_word_len
        self.custom_stopwords = set(s.strip().lower() for s in (custom_stopwords or []) if s.strip())
        self.stopwords = DEFAULT_STOPWORDS.union(self.custom_stopwords)
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.vectorizer: Optional[CountVectorizer] = None

    def clean_text(self, text: str) -> str:
        """Clean an individual text string."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Lowercase
        text = text.lower()

        # Remove URLs and emails
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", " ", text)

        # Remove numbers if configured
        if self.remove_numbers:
            text = re.sub(r"\b\d+\b", " ", text)
            text = re.sub(r"\d+", " ", text)

        # Keep only alphabetic characters and whitespace
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # Tokenize simply by splitting
        tokens = text.split()

        # Filter by min word length and stopwords
        filtered_tokens = []
        for token in tokens:
            if len(token) < self.min_word_len:
                continue
            if token in self.stopwords:
                continue
            if self.apply_lemmatization and LEMMATIZER is not None:
                try:
                    token = LEMMATIZER.lemmatize(token)
                except Exception:
                    pass
            if token not in self.stopwords:
                filtered_tokens.append(token)

        return " ".join(filtered_tokens)

    def preprocess_corpus(self, documents: List[str]) -> List[str]:
        """Clean a list of documents."""
        return [self.clean_text(doc) for doc in documents]

    def fit_transform(self, documents: List[str]):
        """Clean and fit CountVectorizer on documents, returning (dtm, feature_names)."""
        cleaned_docs = self.preprocess_corpus(documents)

        # In case cleaned_docs contains empty strings, keep placeholder to preserve shape
        cleaned_docs_non_empty = [doc if doc.strip() else "empty_document" for doc in cleaned_docs]

        self.vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            token_pattern=r"(?u)\b\w+\b",
        )

        dtm = self.vectorizer.fit_transform(cleaned_docs_non_empty)
        feature_names = self.vectorizer.get_feature_names_out()
        return dtm, feature_names, cleaned_docs

    def transform(self, documents: List[str]):
        """Transform new documents using the already-fitted vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Preprocessor vectorizer has not been fitted yet.")
        cleaned_docs = self.preprocess_corpus(documents)
        cleaned_docs_non_empty = [doc if doc.strip() else "empty_document" for doc in cleaned_docs]
        return self.vectorizer.transform(cleaned_docs_non_empty), cleaned_docs
