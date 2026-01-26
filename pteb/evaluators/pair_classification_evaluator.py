"""
Pair Classification evaluator for PTEB.

This module implements the PairClassificationEvaluator class that computes
classification metrics for pair classification tasks following the MTEB implementation.
"""

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from .base_evaluator import BaseEvaluator


class PairClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for Pair Classification tasks.

    Computes Average Precision (AP) on embedding similarities.
    Follows the MTEB PairClassificationEvaluator implementation.
    """

    def __init__(self, similarity_metrics: list[str] | None = None, **kwargs):
        """
        Initialize the Pair Classification evaluator.

        Args:
            similarity_metrics: List of similarity metrics to compute
                                ("cosine", "dot", "euclidean", "manhattan")
            **kwargs: Additional evaluator parameters
        """
        super().__init__(**kwargs)

        if similarity_metrics is None:
            similarity_metrics = ["cosine", "dot", "euclidean", "manhattan"]

        self.similarity_metrics = similarity_metrics

    @classmethod
    def get_main_score_name(cls) -> str:
        """
        Get the name of the main score for pair classification tasks.

        Returns:
            The name of the main score metric.
        """
        return "ap"

    def __call__(
        self,
        model: Any,
        *,
        encode_kwargs: dict[str, Any] | None = None,
        sentences1: list[str] | None = None,
        sentences2: list[str] | None = None,
        labels: list[int] | None = None,
        phase: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate the embedding model on pair classification data.

        Following MTEB pattern with flexible parameter passing.

        Args:
            model: The embedding model to evaluate
            encode_kwargs: Keyword arguments to pass to the model's encode method
            sentences1: List of first sentences
            sentences2: List of second sentences
            labels: List of binary labels (0 or 1)
            phase: Optional phase identifier for logging
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing classification metrics and raw data for MAE calculation
        """
        # Validate required inputs
        if not (sentences1 and sentences2 and labels):
            raise ValueError("sentences1, sentences2, and labels are required")

        # Validate inputs
        self._validate_inputs(sentences1, sentences2, labels)
        self._validate_labels(labels)

        # Ensure model is on GPU before evaluation
        self._ensure_gpu(model)

        # Get unique sentences to avoid encoding duplicates
        all_sentences = sentences1 + sentences2
        unique_sentences = list(
            dict.fromkeys(all_sentences)
        )  # Preserves order while removing duplicates

        # Encode all unique sentences in one batch
        step_prefix = f"{phase} " if phase else ""
        print(
            f"{step_prefix}Encoding {len(unique_sentences)} unique sentences (extracted from {len(sentences1)} pairs)"
        )
        unique_embeddings = self._encode_sentences(
            unique_sentences,
            model,
            emb_model_name=f"{step_prefix}Encoding",
            encode_kwargs=encode_kwargs,
            task_name="PairClassification",
        )

        # Check for OOM errors during encoding
        if unique_embeddings is None:
            print("ERROR: CUDA OOM occurred during encoding - returning error result")
            return {
                "error": "CUDA_OOM",
                "error_message": "CUDA out of memory during sentence encoding",
                "accuracy": 0.0,
                "f1": 0.0,
                "ap": 0.0,
            }

        # Create mapping from sentences to embeddings
        sentence_to_embedding = {
            sentence: embedding
            for sentence, embedding in zip(unique_sentences, unique_embeddings)
        }

        # Extract embeddings for sentences1 and sentences2 using the mapping
        embeddings1 = np.array([sentence_to_embedding[sent] for sent in sentences1])
        embeddings2 = np.array([sentence_to_embedding[sent] for sent in sentences2])

        results = {}
        primary_similarities = None

        # Compute metrics for each similarity metric
        for metric in self.similarity_metrics:
            # Compute similarities
            similarities = self._compute_similarities(embeddings1, embeddings2, metric)

            # Store primary similarities for MAE calculation (use cosine if available, otherwise first metric)
            if metric == "cosine" or primary_similarities is None:
                primary_similarities = similarities

            # Compute all classification metrics
            metric_results = self._compute_classification_metrics(similarities, labels)

            # Store results with metric prefix
            for key, value in metric_results.items():
                results[f"{metric}_{key}"] = value

        return results

    def _validate_inputs(
        self, sentences1: list[str], sentences2: list[str], labels: list[int]
    ) -> None:
        """
        Validate that input lists have consistent lengths.

        Args:
            sentences1: List of first sentences
            sentences2: List of second sentences
            labels: List of binary labels

        Raises:
            ValueError: If input lengths don't match
        """
        if len(sentences1) != len(sentences2):
            raise ValueError(
                f"Sentence lists must have same length: {len(sentences1)} vs {len(sentences2)}"
            )

        if len(sentences1) != len(labels):
            raise ValueError(
                f"Sentences and labels must have same length: {len(sentences1)} vs {len(labels)}"
            )

        if len(sentences1) == 0:
            raise ValueError("Input lists cannot be empty")

    def _validate_labels(self, labels: list[int]) -> None:
        """
        Validate that labels are binary (0 or 1).

        Args:
            labels: List of labels

        Raises:
            ValueError: If labels are not binary
        """
        unique_labels = set(labels)
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels must be binary (0 or 1), got: {unique_labels}")

    def _compute_classification_metrics(
        self, similarities: np.ndarray, labels: list[int]
    ) -> dict[str, Any]:
        """
        Compute Average Precision for given similarities and labels.

        Args:
            similarities: Array of computed similarities
            labels: List of ground truth binary labels

        Returns:
            Dictionary containing AP score
        """
        ap = average_precision_score(labels, similarities)
        return {"ap": ap}

    def __str__(self) -> str:
        """String representation of the evaluator."""
        return f"PairClassificationEvaluator(metrics={self.similarity_metrics})"
