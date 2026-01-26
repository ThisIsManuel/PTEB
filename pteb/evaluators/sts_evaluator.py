"""
STS (Semantic Textual Similarity) evaluator for PTEB.

This module implements the STSEvaluator class that computes correlation metrics
between embedding similarities and ground truth scores.
"""

from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .base_evaluator import BaseEvaluator


class STSEvaluator(BaseEvaluator):
    """
    Evaluator for Semantic Textual Similarity tasks.

    Computes Spearman and Pearson correlations between embedding-based
    similarities and ground truth similarity scores.
    """

    def __init__(
        self,
        similarity_metrics: list[str] | None = None,
        dataset_name: str | None = None,
        **kwargs,
    ):
        """
        Initialize the STS evaluator.

        Args:
            similarity_metrics: List of similarity metrics to compute
                                ("cosine", "dot", "euclidean", "manhattan")
            dataset_name: Name of the dataset for score normalization
            **kwargs: Additional evaluator parameters
        """
        super().__init__(**kwargs)

        if similarity_metrics is None:
            similarity_metrics = ["cosine"]

        self.similarity_metrics = similarity_metrics
        self.dataset_name = dataset_name

    @classmethod
    def get_main_score_name(cls) -> str:
        """
        Get the name of the main score for STS tasks.

        Returns:
            The name of the main score metric.
        """
        return "spearman"

    def __call__(
        self,
        model: Any,
        *,
        encode_kwargs: dict[str, Any] | None = None,
        sentences1: list[str] | None = None,
        sentences2: list[str] | None = None,
        scores: list[float] | None = None,
        phase: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate the embedding model on STS data.

        Following MTEB pattern with flexible parameter passing.

        Args:
            model: The embedding model to evaluate
            encode_kwargs: Keyword arguments to pass to the model's encode method
            sentences1: List of first sentences
            sentences2: List of second sentences
            scores: List of ground truth similarity scores
            phase: Optional phase identifier for logging
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing correlation metrics and raw data for MAE calculation
        """
        # Validate required inputs
        if sentences1 is None or sentences2 is None or scores is None:
            raise ValueError("sentences1, sentences2, and scores are required")

        # Validate inputs
        self._validate_inputs(sentences1, sentences2, scores)

        encode_kwargs = encode_kwargs or {}

        # Ensure model is on GPU before evaluation
        self._ensure_gpu(model)

        # Encode sentences
        step_prefix = f"{phase} " if phase else ""
        embeddings1 = self._encode_sentences(
            sentences1,
            model,
            emb_model_name=f"{step_prefix}1/2",
            encode_kwargs=encode_kwargs,
            task_name="STS",
        )
        embeddings2 = self._encode_sentences(
            sentences2,
            model,
            emb_model_name=f"{step_prefix}2/2",
            encode_kwargs=encode_kwargs,
            task_name="STS",
        )

        # Check for OOM errors during encoding
        if embeddings1 is None or embeddings2 is None:
            print("ERROR: CUDA OOM occurred during encoding - returning error result")
            return {
                "error": "CUDA_OOM",
                "error_message": "CUDA out of memory during sentence encoding",
                "spearman": 0.0,
                "pearson": 0.0,
            }

        # generate results dict
        results = {}
        similarities = None

        # First try model-specific similarity for top-level keys
        if hasattr(model, "similarity_pairwise"):
            try:
                print("Using model-specific similarity method")
                similarities = model.similarity_pairwise(embeddings1, embeddings2)
                similarities = np.array(similarities)

            except Exception as e:
                print(f"Model-specific similarity failed ({e}), falling back to cosine")

        metric = "cosine"
        if similarities is None:
            print("No model-specific similarities calculated... use generic method")
            similarities = self._compute_similarities(
                embeddings1, embeddings2, metric, model
            )

        # Compute correlations
        spearman_corr = self._compute_spearman_correlation(similarities, scores)

        # Store results with MTEB key format
        results[f"{metric}_spearman"] = spearman_corr
        return results

    def _validate_inputs(
        self, sentences1: list[str], sentences2: list[str], scores: list[float]
    ) -> None:
        """
        Validate that input lists have consistent lengths.

        Args:
            sentences1: List of first sentences
            sentences2: List of second sentences
            scores: List of similarity scores

        Raises:
            ValueError: If input lengths don't match
        """
        if len(sentences1) != len(sentences2):
            raise ValueError(
                f"Sentence lists must have same length: {len(sentences1)} vs {len(sentences2)}"
            )

        if len(sentences1) != len(scores):
            raise ValueError(
                f"Sentences and scores must have same length: {len(sentences1)} vs {len(scores)}"
            )

        if len(sentences1) == 0:
            raise ValueError("Input lists cannot be empty")

    def _compute_spearman_correlation(
        self, similarities: np.ndarray, scores: list[float]
    ) -> float:
        """
        Compute Spearman rank correlation.

        Args:
            similarities: Array of computed similarities
            scores: List of ground truth scores

        Returns:
            Spearman correlation coefficient (raw value in [-1, 1])
        """
        try:
            correlation, _ = spearmanr(list(similarities), list(scores))
            # Handle NaN case (e.g., when all similarities are identical)
            # Return raw correlation coefficient for MTEB compatibility
            return float(correlation) if not np.isnan(correlation) else 0.0
        except (ValueError, TypeError, RuntimeError):
            return np.nan

    def __str__(self) -> str:
        """String representation of the evaluator."""
        return f"STSEvaluator(metrics={self.similarity_metrics})"
