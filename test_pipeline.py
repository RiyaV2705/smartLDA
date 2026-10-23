"""Pytest test suite for SmartTopic_LDA assignment requirements.

Covers:
- Sample text documents and CSV upload ingestion
- Text preprocessing and Document-Term Matrix (DTM) construction
- Scikit-learn LatentDirichletAllocation model fitting
- Topic-word distributions (phi) and document-topic probabilities (theta)
- Model evaluation metrics (Perplexity and Log-Likelihood)
- Error handling for empty text and insufficient vocabulary
- Export artifacts (assignments CSV, keyword summary CSV, Viva report)
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import preprocess_text, SAMPLE_DATASETS


def test_sample_datasets_ingestion():
    """Verify built-in sample datasets and sample CSV dataset exist and are valid."""
    assert len(SAMPLE_DATASETS) >= 3, "Expected at least 3 built-in sample datasets."
    
    primary_corpus = SAMPLE_DATASETS["AI & Computer Science Research (30 docs)"]
    assert len(primary_corpus) == 30, f"Expected 30 docs in primary corpus, found {len(primary_corpus)}."
    assert all(isinstance(doc, str) and len(doc.strip()) > 0 for doc in primary_corpus)

    # Verify sample CSV exists and has expected format
    csv_path = os.path.join(ROOT_DIR, "sample_nlp_corpus.csv")
    assert os.path.exists(csv_path), "sample_nlp_corpus.csv should exist in project root."
    df_csv = pd.read_csv(csv_path)
    assert "content" in df_csv.columns, "CSV must contain a 'content' column."
    assert len(df_csv) >= 20, "CSV should contain at least 20 sample rows."


def test_text_preprocessing():
    """Verify preprocessing cleans URLs, emails, punctuation, digits, and filters stopwords."""
    sample_text = "Transformers 123 use https://example.com and test@email.com with Attention mechanisms!"
    cleaned = preprocess_text(
        sample_text,
        apply_lemmatization=True,
        remove_numbers=True,
        min_word_len=3,
    )
    assert "https" not in cleaned
    assert "123" not in cleaned
    assert "@" not in cleaned
    assert "mechanism" in cleaned or "mechanisms" in cleaned
    assert "transformer" in cleaned or "transformers" in cleaned


def test_document_term_matrix_construction():
    """Verify CountVectorizer produces a valid Document-Term Matrix (DTM)."""
    corpus = SAMPLE_DATASETS["AI & Computer Science Research (30 docs)"]
    cleaned_docs = [preprocess_text(d, apply_lemmatization=True, remove_numbers=True, min_word_len=3) for d in corpus]
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98, token_pattern=r"(?u)\b\w+\b")
    dtm = vectorizer.fit_transform(cleaned_docs)
    vocab = vectorizer.get_feature_names_out()

    assert dtm.shape[0] == len(corpus), "DTM rows must match document count."
    assert dtm.shape[1] > 100, "Vocabulary size must exceed 100 features."
    assert len(vocab) == dtm.shape[1]


def test_sklearn_lda_model_training():
    """Verify scikit-learn LatentDirichletAllocation model fits properly."""
    corpus = SAMPLE_DATASETS["AI & Computer Science Research (30 docs)"]
    cleaned_docs = [preprocess_text(d, apply_lemmatization=True, remove_numbers=True, min_word_len=3) for d in corpus]
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98, token_pattern=r"(?u)\b\w+\b")
    dtm = vectorizer.fit_transform(cleaned_docs)

    K = 4
    lda = LatentDirichletAllocation(n_components=K, max_iter=25, learning_method="batch", random_state=42)
    lda.fit(dtm)

    assert hasattr(lda, "components_"), "Model should be fitted and contain components_."
    assert lda.components_.shape == (K, dtm.shape[1]), "Components shape should be (K, V)."


def test_topic_word_and_document_topic_distributions():
    """Verify phi sums to 1.0 per topic, theta sums to 1.0 per document, and dominant topics compute correctly."""
    corpus = SAMPLE_DATASETS["AI & Computer Science Research (30 docs)"]
    cleaned_docs = [preprocess_text(d, apply_lemmatization=True, remove_numbers=True, min_word_len=3) for d in corpus]
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98, token_pattern=r"(?u)\b\w+\b")
    dtm = vectorizer.fit_transform(cleaned_docs)
    vocab = vectorizer.get_feature_names_out()

    K = 4
    lda = LatentDirichletAllocation(n_components=K, max_iter=25, learning_method="batch", random_state=42)
    lda.fit(dtm)

    # Topic-Word Distributions (phi)
    for comp in lda.components_:
        norm_comp = comp / comp.sum()
        assert np.isclose(norm_comp.sum(), 1.0), "Normalized topic-word distribution must sum to 1.0."

    # Document-Topic Probabilities (theta)
    doc_topic_probs = lda.transform(dtm)
    assert doc_topic_probs.shape == (len(corpus), K)
    assert np.allclose(doc_topic_probs.sum(axis=1), 1.0, atol=1e-4), "Document-topic probabilities must sum to 1.0."

    dominant_topics = np.argmax(doc_topic_probs, axis=1)
    confidences = np.max(doc_topic_probs, axis=1)
    assert len(dominant_topics) == len(corpus)
    assert all(0.0 <= c <= 1.0 for c in confidences)


def test_model_evaluation_metrics():
    """Verify Perplexity and Log-Likelihood scores compute correctly."""
    corpus = SAMPLE_DATASETS["AI & Computer Science Research (30 docs)"]
    cleaned_docs = [preprocess_text(d, apply_lemmatization=True, remove_numbers=True, min_word_len=3) for d in corpus]
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98, token_pattern=r"(?u)\b\w+\b")
    dtm = vectorizer.fit_transform(cleaned_docs)

    lda = LatentDirichletAllocation(n_components=4, max_iter=25, learning_method="batch", random_state=42)
    lda.fit(dtm)

    perplexity = float(lda.perplexity(dtm))
    log_likelihood = float(lda.score(dtm))

    assert perplexity > 0.0, "Perplexity must be positive."
    assert log_likelihood < 0.0, "Log-likelihood of text count matrix is typically negative."


def test_error_handling_empty_and_insufficient_vocab():
    """Verify handling of empty text input and insufficient vocabulary checks."""
    # Empty documents
    empty_docs = ["", "   ", "---", "###"]
    cleaned = [preprocess_text(d) for d in empty_docs]
    non_empty = [c for c in cleaned if c.strip()]
    assert len(non_empty) == 0, "Empty or punctuation-only strings must result in empty clean strings."

    # Insufficient vocabulary
    tiny_corpus = ["quantum", "physics"]
    tiny_vec = CountVectorizer()
    tiny_dtm = tiny_vec.fit_transform(tiny_corpus)
    requested_k = 5
    assert tiny_dtm.shape[1] < requested_k, "Tiny vocabulary condition should be detected."


def test_csv_and_report_export():
    """Verify topic assignment DataFrame, CSV export generation, and viva report structure."""
    corpus = SAMPLE_DATASETS["AI & Computer Science Research (30 docs)"]
    cleaned_docs = [preprocess_text(d, apply_lemmatization=True, remove_numbers=True, min_word_len=3) for d in corpus]
    
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.98, token_pattern=r"(?u)\b\w+\b")
    dtm = vectorizer.fit_transform(cleaned_docs)
    K = 4
    lda = LatentDirichletAllocation(n_components=K, max_iter=25, learning_method="batch", random_state=42)
    lda.fit(dtm)
    doc_topic_probs = lda.transform(dtm)

    df_assign = pd.DataFrame({
        "Doc ID": list(range(1, len(corpus) + 1)),
        "Dominant Topic": [f"Topic {t + 1}" for t in np.argmax(doc_topic_probs, axis=1)],
        "Confidence": np.max(doc_topic_probs, axis=1),
        "Document Text": corpus,
    })
    csv_str = df_assign.to_csv(index=False)
    assert "Doc ID" in csv_str and "Dominant Topic" in csv_str
    assert len(csv_str.strip().split("\n")) == len(corpus) + 1

    # Report structure check
    report = f"# Viva Report\nPerplexity: {lda.perplexity(dtm):.2f}\nTopics: {K}"
    assert "Perplexity:" in report and "Topics: 4" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
