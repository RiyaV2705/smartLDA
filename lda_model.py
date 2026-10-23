"""LDA Topic Modeling training, evaluation, and inference wrapper."""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from typing import Dict, List, Tuple, Any, Optional
import joblib


class LDAModelTrainer:
    """Trains and manages Latent Dirichlet Allocation (LDA) models."""

    def __init__(
        self,
        n_topics: int = 5,
        max_iter: int = 25,
        learning_method: str = "batch",
        random_state: int = 42,
        doc_topic_prior: Optional[float] = None,
        topic_word_prior: Optional[float] = None,
    ):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.learning_method = learning_method
        self.random_state = random_state
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior

        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            learning_method=self.learning_method,
            random_state=self.random_state,
            doc_topic_prior=self.doc_topic_prior,
            topic_word_prior=self.topic_word_prior,
            n_jobs=-1,
        )
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.topic_names: Dict[int, str] = {}

    def fit(self, dtm, feature_names: List[str]):
        """Fit LDA model on Document-Term Matrix."""
        self.feature_names = list(feature_names)
        self.model.fit(dtm)
        self.is_fitted = True

        # Auto-generate descriptive topic names from top 3 words
        self.topic_names = {}
        for topic_idx, topic in enumerate(self.model.components_):
            top_word_indices = topic.argsort()[: -4 : -1]
            top_words = [self.feature_names[i] for i in top_word_indices]
            self.topic_names[topic_idx] = f"Topic {topic_idx + 1}: {', '.join(top_words)}"

        return self

    def set_custom_topic_name(self, topic_idx: int, name: str):
        """Allow user to provide custom label for a topic."""
        self.topic_names[topic_idx] = name

    def get_top_words(self, n_top_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """Return the top words and their normalized weights for each topic."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        topics_dict = {}
        for topic_idx, topic in enumerate(self.model.components_):
            total_weight = topic.sum()
            norm_topic = topic / total_weight if total_weight > 0 else topic
            top_indices = norm_topic.argsort()[: -n_top_words - 1 : -1]

            words_weights = [
                (self.feature_names[i], float(norm_topic[i]))
                for i in top_indices
            ]
            topics_dict[topic_idx] = words_weights
        return topics_dict

    def transform(self, dtm) -> np.ndarray:
        """Transform documents to document-topic probability distributions."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.transform(dtm)

    def evaluate(self, dtm) -> Dict[str, float]:
        """Compute model perplexity and log-likelihood metrics."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        perplexity = float(self.model.perplexity(dtm))
        log_likelihood = float(self.model.score(dtm))
        return {
            "perplexity": round(perplexity, 2),
            "log_likelihood": round(log_likelihood, 2),
            "n_topics": self.n_topics,
            "vocab_size": len(self.feature_names),
        }

    def get_document_topic_dataframe(
        self, raw_documents: List[str], cleaned_documents: List[str], doc_topic_dist: np.ndarray
    ) -> pd.DataFrame:
        """Build a comprehensive DataFrame showing dominant topic and probabilities for each doc."""
        dominant_topics = np.argmax(doc_topic_dist, axis=1)
        confidences = np.max(doc_topic_dist, axis=1)

        data = {
            "Doc ID": list(range(1, len(raw_documents) + 1)),
            "Dominant Topic Index": dominant_topics,
            "Dominant Topic Label": [self.topic_names.get(t, f"Topic {t + 1}") for t in dominant_topics],
            "Confidence": [round(float(c), 4) for c in confidences],
            "Original Text": raw_documents,
            "Cleaned Text": cleaned_documents,
        }

        # Add individual topic probability columns
        for t_idx in range(self.n_topics):
            label = self.topic_names.get(t_idx, f"Topic {t_idx + 1}")
            data[f"P({label})"] = [round(float(p), 4) for p in doc_topic_dist[:, t_idx]]

        return pd.DataFrame(data)

    def predict_single(self, preprocessor, text: str) -> Dict[str, Any]:
        """Predict topic distribution for a new unseen text string."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        dtm_new, cleaned = preprocessor.transform([text])
        probs = self.model.transform(dtm_new)[0]
        dominant_idx = int(np.argmax(probs))

        topic_breakdown = []
        for idx, prob in enumerate(probs):
            topic_breakdown.append({
                "topic_index": idx,
                "topic_label": self.topic_names.get(idx, f"Topic {idx + 1}"),
                "probability": float(prob),
                "top_words": [w for w, _ in self.get_top_words(5)[idx]],
            })

        topic_breakdown.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "cleaned_text": cleaned[0],
            "dominant_topic_index": dominant_idx,
            "dominant_topic_label": self.topic_names.get(dominant_idx, f"Topic {dominant_idx + 1}"),
            "confidence": float(probs[dominant_idx]),
            "breakdown": topic_breakdown,
        }

    def save(self, filepath: str):
        """Save model and metadata to disk."""
        data = {
            "model": self.model,
            "n_topics": self.n_topics,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "topic_names": self.topic_names,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(data, filepath)

    @classmethod
    def load(cls, filepath: str) -> "LDAModelTrainer":
        """Load model and metadata from disk."""
        data = joblib.load(filepath)
        trainer = cls(
            n_topics=data["n_topics"],
            max_iter=data["max_iter"],
            random_state=data["random_state"],
        )
        trainer.model = data["model"]
        trainer.feature_names = data["feature_names"]
        trainer.topic_names = data["topic_names"]
        trainer.is_fitted = data["is_fitted"]
        return trainer
