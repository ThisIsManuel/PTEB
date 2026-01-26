"""
Clustering evaluator for PTEB.

This module implements the ClusteringEvaluator class that computes
clustering metrics using Mini-Batch K-Means following the MTEB implementation.
"""

import logging
from typing import Any

import sklearn
import sklearn.cluster
from sklearn import metrics

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class ClusteringEvaluator(BaseEvaluator):
    """
    Evaluator for Clustering tasks.

    Uses Mini-Batch K-Means clustering to group sentences and evaluates
    using v-measure score. Follows the MTEB ClusteringEvaluator implementation.
    """

    def __init__(
        self,
        clustering_batch_size: int = 32,
        task_name: str | None = None,
        **kwargs,
    ):
        """
        Initialize the Clustering evaluator.

        Args:
            clustering_batch_size: Batch size for Mini-Batch K-Means algorithm.
                                 This controls memory usage and speed, not accuracy.
                                 MTEB uses 500 by default. Separate from embedding batch size.
            task_name: Name of the task for logging
            **kwargs: Additional evaluator parameters (including emb_batch_size for encoding)
        """
        super().__init__(**kwargs)

        self.clustering_batch_size = clustering_batch_size
        self.task_name = task_name

    @classmethod
    def get_main_score_name(cls) -> str:
        """
        Get the name of the main score for clustering tasks.

        Returns:
            The name of the main score metric.
        """
        return "v_measure"

    def __call__(
        self,
        model: Any,
        *,
        encode_kwargs: dict[str, Any] | None = None,
        sentences: list[str] | None = None,
        labels: list[str] | None = None,
        phase: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate the embedding model on clustering data.

        Following MTEB pattern with flexible parameter passing.

        Args:
            model: The embedding model to evaluate
            encode_kwargs: Keyword arguments to pass to the model's encode method
            sentences: List of sentences to cluster
            labels: List of cluster labels (ground truth)
            phase: Optional phase identifier for logging
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing clustering metrics
        """
        # Validate required inputs
        if sentences is None or labels is None:
            raise ValueError("sentences and labels are required")

        # Validate inputs
        self._validate_inputs(sentences, labels)

        # Ensure model is on GPU before evaluation
        self._ensure_gpu(model)

        encode_kwargs = encode_kwargs or {}

        # Set default embedding batch size if not provided (for encoding)
        # Note: This is separate from clustering_batch_size (for K-Means)
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 512  # MTEB default for encoding

        # Encode sentences
        step_prefix = f"{phase} " if phase else ""
        logger.info(f"Encoding {len(sentences)} sentences...")
        print(f"{step_prefix}Encoding {len(sentences)} sentences for clustering...")

        corpus_embeddings = self._encode_sentences(
            sentences,
            model,
            emb_model_name=f"{step_prefix}Clustering",
            encode_kwargs=encode_kwargs,
            task_name="Clustering",
        )

        # Check for OOM errors during encoding
        if corpus_embeddings is None:
            print("ERROR: CUDA OOM occurred during encoding - returning error result")
            return {
                "error": "CUDA_OOM",
                "error_message": "CUDA out of memory during sentence encoding",
                "v_measure": 0.0,
            }

        # Fit Mini-Batch K-Means model
        n_clusters = len(set(labels))
        logger.info(f"Fitting Mini-Batch K-Means model with {n_clusters} clusters...")
        print(
            f"{step_prefix}Fitting Mini-Batch K-Means model with {n_clusters} clusters..."
        )

        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=self.clustering_batch_size,
            n_init="auto",
            random_state=self.seed,
        )

        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        # Evaluate clustering performance
        logger.info("Evaluating clustering performance...")
        print(f"{step_prefix}Evaluating clustering performance...")

        v_measure = metrics.cluster.v_measure_score(labels, cluster_assignment)

        results = {
            "v_measure": v_measure,
        }

        print(
            f"{step_prefix} Clustering evaluation completed. V-measure: {v_measure:.4f}"
        )

        return results

    def _validate_inputs(self, sentences: list[str], labels: list[str]) -> None:
        """
        Validate clustering inputs.

        Args:
            sentences: List of sentences
            labels: List of cluster labels

        Raises:
            ValueError: If inputs are invalid
        """
        if len(sentences) != len(labels):
            raise ValueError(
                f"Sentences and labels must have same length: {len(sentences)} vs {len(labels)}"
            )

        if len(sentences) == 0:
            raise ValueError("Input lists cannot be empty")

        # Check that we have at least 2 unique labels for clustering
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            raise ValueError(
                f"Need at least 2 unique labels for clustering, got {len(unique_labels)}"
            )

        print(
            f"Clustering validation passed: {len(sentences)} sentences, {len(unique_labels)} unique labels"
        )
