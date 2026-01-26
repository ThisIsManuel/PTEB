"""
Summarization evaluator for PTEB.

This module implements the SummarizationEvaluator class that evaluates
machine-generated summaries by comparing their embeddings with human-written
summaries and computing correlations with human assessment scores.

Based on MTEB's SummarizationEvaluator with adaptations for PTEB architecture.
"""

import logging

# Compute global correlations across all samples
import math
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sentence_transformers.util import cos_sim, dot_score

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class SummarizationEvaluator(BaseEvaluator):
    """
    Evaluator for summarization tasks.

    Evaluates machine-generated summaries by comparing their embeddings with
    human-written summaries and computing correlations with human assessment scores.

    The evaluator:
    1. Encodes human and machine summaries using the embedding model
    2. Computes similarity between machine summaries and human summaries
    3. Takes the maximum similarity for each machine summary
    4. Calculates correlations between predicted and gold assessment scores

    Following MTEB's methodology: Spearman correlation based on cosine similarity
    serves as the main metric.
    """

    def __init__(
        self,
        similarity_metrics: list[str] | None = None,
        task_name: str | None = None,
        **kwargs,
    ):
        """
        Initialize the summarization evaluator.

        Args:
            similarity_metrics: List of similarity metrics to compute
                              ("cosine", "dot", "model_specific")
            task_name: Name of the task for model encoding
            **kwargs: Additional evaluator parameters
        """
        super().__init__(**kwargs)

        if similarity_metrics is None:
            similarity_metrics = ["cosine"]

        self.similarity_metrics = similarity_metrics
        self.task_name = task_name

    @classmethod
    def get_main_score_name(cls) -> str:
        """
        Get the name of the main score for summarization tasks.

        Returns:
            The name of the main score metric.
        """
        return "spearman"

    def __call__(
        self,
        model: Any,
        *,
        encode_kwargs: dict[str, Any] | None = None,
        human_summaries: list[list[str]] | None = None,
        machine_summaries: list[list[str]] | None = None,
        gold_scores: list[list[float]] | None = None,
        texts: list[str] | None = None,
        phase: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate the embedding model on summarization data.

        Following MTEB pattern with flexible parameter passing.

        Args:
            model: The embedding model to evaluate
            encode_kwargs: Keyword arguments to pass to the model's encode method
            human_summaries: List of lists of human-written summaries
                           Shape: (n_samples, n_human_summaries_per_sample)
            machine_summaries: List of lists of machine-generated summaries
                             Shape: (n_samples, n_machine_summaries_per_sample)
            gold_scores: List of lists of human assessment scores for machine summaries
                        Shape: (n_samples, n_machine_summaries_per_sample)
            texts: Optional list of original texts that were summarized
            phase: Optional phase identifier for logging
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing correlation metrics
        """
        # Avoid mutable default
        if encode_kwargs is None:
            encode_kwargs = {}

        # Validate required inputs
        if human_summaries is None or machine_summaries is None or gold_scores is None:
            raise ValueError(
                "human_summaries, machine_summaries, and gold_scores are required"
            )

        # Validate inputs
        self._validate_inputs(
            human_summaries, machine_summaries, gold_scores
        )

        # Ensure model is on GPU before evaluation
        self._ensure_gpu(model)

        # Prepare data for encoding
        human_lens = [len(summaries) for summaries in human_summaries]
        machine_lens = [len(summaries) for summaries in machine_summaries]

        step_prefix = f"{phase} " if phase else ""

        # Flatten and encode all human summaries
        logger.info("%sEncoding human summaries...", step_prefix)
        all_human_summaries = [
            summary for summaries in human_summaries for summary in summaries
        ]
        embs_human_all = self._encode_sentences(
            all_human_summaries,
            model,
            emb_model_name=f"{step_prefix}Human 1/2",
            encode_kwargs=encode_kwargs,
            task_name="Summarization",
        )

        # Flatten and encode all machine summaries
        logger.info("%sEncoding machine summaries...", step_prefix)
        all_machine_summaries = [
            summary for summaries in machine_summaries for summary in summaries
        ]
        embs_machine_all = self._encode_sentences(
            all_machine_summaries,
            model,
            emb_model_name=f"{step_prefix}Machine 2/2",
            encode_kwargs=encode_kwargs,
            task_name="Summarization",
        )

        # Check for OOM errors during encoding
        if embs_human_all is None or embs_machine_all is None:
            print("ERROR: CUDA OOM occurred during encoding - returning error result")
            return {
                "error": "CUDA_OOM",
                "error_message": "CUDA out of memory during sentence encoding",
                "cos_sim": {"spearman": 0.0, "pearson": 0.0},
                "dot": {"spearman": 0.0, "pearson": 0.0},
            }

        # Split embeddings back into per-sample groups
        embs_human_grouped = self._split_embeddings(embs_human_all, human_lens)
        embs_machine_grouped = self._split_embeddings(embs_machine_all, machine_lens)

        # Compute correlations for each similarity metric
        results = {}
        primary_correlations = None

        for metric in self.similarity_metrics:
            correlations = self._compute_summarization_correlations(
                embs_human_grouped,
                embs_machine_grouped,
                gold_scores,
                metric,
                model,
                step_prefix,
            )

            # Store primary correlations (cosine if available, otherwise first metric)
            if metric == "cosine" or primary_correlations is None:
                primary_correlations = correlations

            # Store results
            if len(self.similarity_metrics) == 1:
                # Single metric - use simple names
                results.update(correlations)
            else:
                # Multiple metrics - prefix with metric name
                for key, value in correlations.items():
                    results[f"{metric}_{key}"] = value

        return results

    def _validate_inputs(
        self,
        human_summaries: list[list[str]],
        machine_summaries: list[list[str]],
        gold_scores: list[list[float]],
    ) -> None:
        """
        Validate summarization input data.

        Args:
            human_summaries: Human-written summaries
            machine_summaries: Machine-generated summaries
            gold_scores: Human assessment scores

        Raises:
            ValueError: If input validation fails
        """
        if len(human_summaries) != len(machine_summaries):
            raise ValueError(
                f"Human and machine summaries must have same length: "
                f"{len(human_summaries)} vs {len(machine_summaries)}"
            )

        if len(human_summaries) != len(gold_scores):
            raise ValueError(
                f"Summaries and scores must have same length: "
                f"{len(human_summaries)} vs {len(gold_scores)}"
            )

        if len(human_summaries) == 0:
            raise ValueError("Input lists cannot be empty")

        # Validate per-sample structure
        for i, (_human_sums, machine_sums, scores) in enumerate(
            zip(human_summaries, machine_summaries, gold_scores)
        ):
            if len(machine_sums) != len(scores):
                raise ValueError(
                    f"Sample {i}: machine summaries and scores must have same length: "
                    f"{len(machine_sums)} vs {len(scores)}"
                )

    def _split_embeddings(
        self, embeddings: np.ndarray, lengths: list[int]
    ) -> list[np.ndarray]:
        """
        Split flat embeddings array back into per-sample groups.

        Args:
            embeddings: Flat array of embeddings
            lengths: Number of embeddings per sample

        Returns:
            List of embedding arrays, one per sample
        """
        if len(lengths) == 0:
            return []

        # Use cumulative sum to get split indices, excluding the last index
        split_indices = np.cumsum(lengths)[:-1]
        return np.split(embeddings, split_indices)

    def _compute_summarization_correlations(
        self,
        embs_human_grouped: list[np.ndarray],
        embs_machine_grouped: list[np.ndarray],
        gold_scores: list[list[float]],
        similarity_metric: str,
        _embedding_model: Any,
        step_prefix: str = "",
    ) -> dict[str, Any]:
        """
        Compute correlations using optimized sentence-transformers batch processing.

        Following proper summarization evaluation methodology:
        1. For each machine summary, compute similarities with ALL human summaries
        2. Take the maximum similarity as the predicted score
        3. Collect all predicted scores and gold scores across all samples
        4. Compute global correlations between predicted scores and gold scores

        Args:
            embs_human_grouped: Human summary embeddings grouped by sample
            embs_machine_grouped: Machine summary embeddings grouped by sample
            gold_scores: Human assessment scores
            similarity_metric: Similarity metric to use
            embedding_model: Embedding model (for model-specific similarity)
            step_prefix: Prefix for logging

        Returns:
            Dictionary with Pearson and Spearman correlation scores
        """
        all_predicted_scores = []
        all_gold_scores = []

        logger.info(
            "%sComputing correlations using optimized %s similarity...",
            step_prefix,
            similarity_metric,
        )

        # torch is a required dependency

        for embs_human, embs_machine, scores in zip(
            embs_human_grouped, embs_machine_grouped, gold_scores
        ):
            if isinstance(embs_human, np.ndarray):
                embs_human_tensor = torch.tensor(embs_human)
            elif isinstance(embs_human, list):
                embs_human_tensor = torch.tensor(np.array(embs_human))
            else:
                embs_human_tensor = embs_human

            if isinstance(embs_machine, np.ndarray):
                embs_machine_tensor = torch.tensor(embs_machine)
            elif isinstance(embs_machine, list):
                embs_machine_tensor = torch.tensor(np.array(embs_machine))
            else:
                embs_machine_tensor = embs_machine

            if similarity_metric == "cosine":
                similarity_matrix = cos_sim(embs_machine_tensor, embs_human_tensor)
                max_similarities = torch.max(similarity_matrix, dim=1)[0]
                predicted_scores = max_similarities.detach().cpu().numpy().tolist()
            elif similarity_metric == "dot":
                similarity_matrix = dot_score(embs_machine_tensor, embs_human_tensor)
                max_similarities = torch.max(similarity_matrix, dim=1)[0]
                predicted_scores = max_similarities.detach().cpu().numpy().tolist()
            else:
                predicted_scores = []
                for emb_machine in embs_machine_tensor:
                    # Compute cosine similarity manually as fallback for custom metrics
                    sims = cos_sim(emb_machine.unsqueeze(0), embs_human_tensor)
                    predicted_scores.append(float(torch.max(sims)))

            all_predicted_scores.extend(predicted_scores)
            all_gold_scores.extend(scores)

        # Skip if we have uniform scores (causes correlation issues)
        if len(set(all_gold_scores)) == 1 or len(set(all_predicted_scores)) == 1:
            logger.warning(
                "%sUniform scores detected, cannot compute correlation", step_prefix
            )
            return {"spearman": 0.0}

        try:
            spearman_corr_val, _ = spearmanr(all_gold_scores, all_predicted_scores)
        except (ValueError, RuntimeError) as e:  # robustness
            logger.warning("%sCorrelation computation failed: %s", step_prefix, e)
            spearman_corr_val = 0.0

        # Coerce to float safely
        if not isinstance(spearman_corr_val, (int, float)):
            try:
                spearman_corr_val = float(spearman_corr_val)
            except (TypeError, ValueError):
                spearman_corr_val = 0.0

        spearman_score = (
            0.0
            if spearman_corr_val is None
            or (isinstance(spearman_corr_val, float) and math.isnan(spearman_corr_val))
            else float(spearman_corr_val)
        )

        # Return correlation in 0-1 range (percentage conversion handled centrally)
        return {
            "spearman": spearman_score,
        }

    def __str__(self) -> str:
        """String representation of the evaluator."""
        return f"SummarizationEvaluator(metrics={self.similarity_metrics})"
