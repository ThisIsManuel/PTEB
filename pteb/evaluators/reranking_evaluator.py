"""
Reranking evaluator for PTEB.

This module implements the RerankingEvaluator class that computes MAP (Mean Average
Precision) using sklearn's average_precision_score.
"""

import logging
from typing import Any

import numpy as np
import torch
from sentence_transformers.util import cos_sim
from sklearn.metrics import average_precision_score

try:
    # Current MTEB location
    from mteb.abstasks.AbsTaskRetrieval import PromptType
except Exception:
    try:
        # Older releases occasionally exposed it here
        from mteb.abstasks.AbsTaskReranking import PromptType
    except Exception:
        # Final fallback
        class PromptType:
            query = "query"
            document = "document"


from .base_evaluator import BaseEvaluator

TensorType = torch.Tensor
logger = logging.getLogger(__name__)


class RerankingEvaluator(BaseEvaluator):
    """
    Evaluator for Reranking tasks.

    Computes MAP (Mean Average Precision) by ranking documents according to their
    relevance to queries using sklearn's average_precision_score.
    """

    def __init__(
        self,
        samples: list[dict[str, Any]] | None = None,
        use_batched_encoding: bool = True,
        task_name: str | None = None,
        encode_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Initialize the Reranking evaluator.

        Args:
            samples: List of samples, each with 'query', 'positive', 'negative' keys
            use_batched_encoding: Whether to use batched encoding
            task_name: Name of the task for prompt specification
            encode_kwargs: Default encode arguments
            **kwargs: Additional evaluator parameters
        """
        super().__init__(**kwargs)
        self.use_batched_encoding = use_batched_encoding
        self.task_name = task_name

        # Store encode_kwargs with default batch size
        self.encode_kwargs = dict(encode_kwargs or {})

        # Initialize samples
        self.samples = samples or []

        # Remove samples with empty positive/negative sets
        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())
        len_before = len(self.samples)
        self.samples = [
            sample
            for sample in self.samples
            if len(sample.get("positive", [])) > 0
            and len(sample.get("negative", [])) > 0
        ]
        len_after = len(self.samples)
        if len_after < len_before:
            print(
                f"    Removed {len_before - len_after} samples with empty positive/negative documents"
            )

    @classmethod
    def get_main_score_name(cls) -> str:
        """
        Get the name of the main score for reranking tasks.

        Returns:
            The name of the main score metric.
        """
        return "map"

    def __call__(
        self,
        model: Any,
        *,
        encode_kwargs: dict[str, Any] | None = None,
        task_name: str | None = None,
        phase: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate the embedding model on reranking data.

        Args:
            model: The embedding model to evaluate
            encode_kwargs: Override encode arguments (merged with stored ones)
            task_name: Override task name for prompt specification
            phase: Optional phase identifier for logging
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing reranking metrics
        """
        self._validate_inputs(self.samples)

        logger.info(f"Evaluating {len(self.samples)} reranking samples")

        # Ensure model is on GPU before evaluation
        self._ensure_gpu(model)

        # Merge encode_kwargs (override stored ones)
        final_encode_kwargs = self.encode_kwargs.copy()
        if encode_kwargs:
            final_encode_kwargs.update(encode_kwargs)

        # Use provided task_name or stored one
        final_task_name = task_name or self.task_name

        # Compute metrics
        results = self._evaluate(
            model, self.samples, final_encode_kwargs, final_task_name, phase
        )

        return results

    def _validate_inputs(
        self, samples: list[dict[str, Any]]
    ) -> None:
        """
        Validate reranking inputs.

        Args:
            samples: List of samples with query, positive, negative keys

        Raises:
            ValueError: If inputs are invalid
        """
        if not samples:
            raise ValueError(
                "No valid samples found (samples must have both positive and negative documents)"
            )

        for i, sample in enumerate(samples):
            if "query" not in sample:
                raise ValueError(f"Sample {i} missing 'query' key")

            if "positive" not in sample or not sample["positive"]:
                raise ValueError(f"Sample {i} missing or empty 'positive' list")

            if "negative" not in sample or not sample["negative"]:
                raise ValueError(f"Sample {i} missing or empty 'negative' list")

    def _evaluate(
        self,
        model: Any,
        samples: list[dict[str, Any]],
        encode_kwargs: dict[str, Any],
        task_name: str | None,
        phase: str,
    ) -> dict[str, Any]:
        """
        Standard reranking evaluation.

        Args:
            model: The embedding model
            samples: List of samples with query, positive, negative
            encode_kwargs: Encoding parameters
            task_name: Task name for prompts
            phase: Phase identifier

        Returns:
            Dictionary of evaluation metrics
        """
        if self.use_batched_encoding:
            return self._evaluate_batched(
                model, samples, encode_kwargs, task_name, phase
            )
        else:
            return self._evaluate_individual(
                model, samples, encode_kwargs, task_name, phase
            )

    def _evaluate_batched(
        self,
        model: Any,
        samples: list[dict[str, Any]],
        encode_kwargs: dict[str, Any],
        task_name: str | None,
        phase: str,
    ) -> dict[str, Any]:
        """
        Batched standard reranking evaluation.
        """

        # Extract and encode all queries
        logger.info("Encoding queries...")

        # Check if we have string or list queries
        first_query = samples[0]["query"]

        if isinstance(first_query, str):
            # Simple string queries
            all_queries = [s["query"] for s in samples]
            query_embeddings = model.encode(
                all_queries,
                task_name=task_name,
                prompt_type=PromptType.query,
                **encode_kwargs,
            )
            query_embeddings = np.asarray(query_embeddings)
            query_lengths = [1] * len(samples)
        elif isinstance(first_query, list):
            # List queries - flatten and deduplicate
            all_queries = []
            query_lengths = []
            for sample in samples:
                query = sample["query"]
                all_queries.extend(query)
                query_lengths.append(len(query))

            # Use deduplication for list queries
            query_embeddings = self._encode_unique_texts(
                all_queries, model, task_name, PromptType.query, encode_kwargs
            )
        else:
            raise ValueError(f"Query must be str or list, got {type(first_query)}")

        # Extract and encode all documents
        logger.info("Encoding documents...")
        all_docs = []
        doc_lengths = []

        for sample in samples:
            docs = sample["positive"] + sample["negative"]
            all_docs.extend(docs)
            doc_lengths.append(len(docs))

        # Encode all documents using unique texts optimization
        doc_embeddings = self._encode_unique_texts(
            all_docs, model, task_name, PromptType.document, encode_kwargs
        )

        # Compute similarities and metrics
        logger.info("Computing similarities and metrics...")
        all_ap_scores = []

        query_idx = 0
        doc_idx = 0

        for i, sample in enumerate(samples):
            # Get query embeddings for this sample
            query_len = query_lengths[i]
            sample_query_embs = query_embeddings[query_idx : query_idx + query_len]
            query_idx += query_len

            # Get document embeddings for this sample
            doc_len = doc_lengths[i]
            sample_doc_embs = doc_embeddings[doc_idx : doc_idx + doc_len]
            doc_idx += doc_len

            # Create relevance labels
            num_pos = len(sample["positive"])
            num_neg = len(sample["negative"])
            is_relevant = [True] * num_pos + [False] * num_neg

            # Compute similarities and metrics for this sample
            self._compute_sample_metrics(
                sample_query_embs,
                sample_doc_embs,
                is_relevant,
                all_ap_scores,
                model,
            )

        # Aggregate results
        return self._aggregate_results(all_ap_scores)

    def _evaluate_individual(
        self,
        model: Any,
        samples: list[dict[str, Any]],
        encode_kwargs: dict[str, Any],
        task_name: str | None,
        phase: str,
    ) -> dict[str, Any]:
        """
        Individual (non-batched) standard reranking evaluation.
        """
        logger.info("Evaluating samples individually...")

        all_ap_scores = []

        for i, sample in enumerate(samples):
            if i % 100 == 0:
                logger.info(f"Processing sample {i + 1}/{len(samples)}")

            query = sample["query"]
            positive = list(sample["positive"])
            negative = list(sample["negative"])

            docs = positive + negative
            is_relevant = [True] * len(positive) + [False] * len(negative)

            # Encode query (ensure it's a list)
            if isinstance(query, str):
                query = [query]

            query_embs = model.encode(
                query,
                task_name=task_name,
                prompt_type=PromptType.query,
                **encode_kwargs,
            )
            query_embs = np.asarray(query_embs)

            # Encode documents
            doc_embs = model.encode(
                docs,
                task_name=task_name,
                prompt_type=PromptType.document,
                **encode_kwargs,
            )
            doc_embs = np.asarray(doc_embs)

            # Compute metrics for this sample
            self._compute_sample_metrics(
                query_embs,
                doc_embs,
                is_relevant,
                all_ap_scores,
                model,
            )

        # Aggregate results
        return self._aggregate_results(all_ap_scores)

    def _compute_sample_metrics(
        self,
        query_embs: np.ndarray,
        doc_embs: np.ndarray,
        is_relevant: list[bool],
        all_ap_scores: list[float],
        model: Any,
    ) -> None:
        """
        Compute metrics for a single sample.

        Args:
            query_embs: Query embeddings
            doc_embs: Document embeddings
            is_relevant: List of relevance labels
            all_ap_scores: List to append AP scores to
            model: The embedding model
        """
        # Compute similarities
        sim_scores = self._compute_sim_scores(query_embs, doc_embs, model)

        # Compute Average Precision
        ap_score = self._compute_average_precision(sim_scores, is_relevant)
        all_ap_scores.append(ap_score)

    def _compute_sim_scores(
        self, query_embs: np.ndarray, doc_embs: np.ndarray, model: Any
    ) -> TensorType | np.ndarray:
        """
        Compute similarity scores between queries and documents.

        Args:
            query_embs: Query embeddings (num_queries, hidden_size)
            doc_embs: Document embeddings (num_docs, hidden_size)
            model: The embedding model

        Returns:
            Similarity scores (num_docs,)
        """
        # Try model-specific similarity method first
        if hasattr(model, "similarity"):
            try:
                sim_scores = model.similarity(query_embs, doc_embs)
                sim_scores = torch.as_tensor(sim_scores)
                if sim_scores.ndim > 1:
                    sim_scores = torch.amax(sim_scores, dim=0)
                return sim_scores
            except Exception as e:
                logger.warning(f"Model-specific similarity failed: {e}")

        # Fall back to cosine similarity
        if cos_sim is not None:
            # Convert to tensors if needed
            if not isinstance(query_embs, torch.Tensor):
                query_embs = torch.tensor(query_embs)
            if not isinstance(doc_embs, torch.Tensor):
                doc_embs = torch.tensor(doc_embs)

            sim_scores = cos_sim(query_embs, doc_embs)
            if len(sim_scores.shape) > 1:
                sim_scores = torch.amax(sim_scores, dim=0)
            return sim_scores
        else:
            # Manual cosine similarity computation
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.preprocessing import normalize

            # Normalize embeddings
            query_norm = normalize(query_embs, norm="l2")
            doc_norm = normalize(doc_embs, norm="l2")

            # Compute similarities
            if query_norm.shape[0] == 1:
                sim_scores = np.sum(query_norm * doc_norm, axis=1)
            else:
                # Multiple queries - take maximum similarity
                sim_matrix = cosine_similarity(query_norm, doc_norm)
                sim_scores = np.max(sim_matrix, axis=0)

            # Convert to tensor for consistency
            return torch.tensor(sim_scores)

    def _compute_average_precision(
        self, sim_scores: TensorType | np.ndarray, is_relevant: list[bool]
    ) -> float:
        """
        Compute Average Precision score.

        Args:
            sim_scores: Similarity scores
            is_relevant: Relevance labels

        Returns:
            AP score
        """
        # Convert to numpy if needed
        if isinstance(sim_scores, torch.Tensor):
            sim_scores = sim_scores.cpu().numpy()

        try:
            return float(average_precision_score(is_relevant, sim_scores))
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Could not compute average precision: {e}")
            return 0.0

    def _aggregate_results(
        self,
        all_ap_scores: list[float],
    ) -> dict[str, Any]:
        """
        Aggregate results across all samples.

        Args:
            all_ap_scores: List of AP scores

        Returns:
            Dictionary of aggregated metrics
        """
        if not all_ap_scores:
            return {"map": 0.0}

        # Main metrics
        mean_ap = float(np.mean(all_ap_scores))

        results = {
            "map": mean_ap,
        }

        return results

    @staticmethod
    def _encode_unique_texts(
        all_texts: list[str],
        model: Any,
        task_name: str | None,
        prompt_type: str,
        encode_kwargs: dict[str, Any],
    ) -> np.ndarray:
        """
        Encode texts with deduplication optimization.

        Args:
            all_texts: List of texts to encode
            model: Embedding model
            task_name: Task name for prompts
            prompt_type: Type of prompt ("query" or "document")
            encode_kwargs: Encoding parameters

        Returns:
            Array of embeddings
        """
        # Create mapping of unique texts
        index_map = {}
        unique_texts = []
        text_indices = []

        # Deduplicate texts while preserving order
        # NB: MTBE uses hashes which can lead to collisions, so we use exact text matching
        for text in all_texts:
            if text not in index_map:
                index_map[text] = len(unique_texts)
            unique_texts.append(text)
            text_indices.append(index_map[text])

        duplicates = len(all_texts) - len(unique_texts)
        if duplicates > 0:
            logger.info(
                f"Found {duplicates}/{len(all_texts)} duplicate texts, encoding unique texts only"
            )

        unique_embeddings = model.encode(
            unique_texts,
            task_name=task_name,
            prompt_type=prompt_type,
            **encode_kwargs,
        )

        unique_embeddings = np.asarray(unique_embeddings)

        # Map back to original order
        return unique_embeddings[text_indices]

    def __str__(self) -> str:
        """String representation of the evaluator."""
        return "RerankingEvaluator(MAP)"
