"""
Reranking task implementation for PTEB.

This module implements the RerankingTask class that handles reranking
dataset loading, paraphrase generation/loading, and evaluation with MAP/MRR metrics.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from sentence_transformers import SentenceTransformer

from ..evaluators.reranking_evaluator import RerankingEvaluator
from ..utils import should_use_multilingual_suffix
from .paraphrase_task import ParaphraseTask

logger = logging.getLogger(__name__)


class RerankingTask(ParaphraseTask):
    """
    Reranking task with paraphrase-enhanced evaluation.

    This class handles:
    - Loading reranking datasets from HuggingFace Hub
    - Generating or loading paraphrases for queries and documents
    - Evaluating embedding models on original and paraphrased data
    - Computing reranking metrics (MAP)
    """

    def __init__(self, datasets: List[str], config: Dict[str, Any], **kwargs):
        """
        Initialize the Reranking task.

        Args:
            datasets: List of reranking dataset names to evaluate on
            config: Configuration dictionary from CLI arguments
            **kwargs: Additional parameters
        """
        # Initialize parent with dataset names
        super().__init__(
            task_name="Reranking",
            task_description="Reranking evaluation with MAP",
            datasets=datasets,
            config=config,
            data_subdir="reranking",  # Store paraphrases in data/paraphrases/reranking/
            **kwargs,
        )

        self.use_batched_encoding = True  # Always enabled for efficiency

        print(f"  Initialized RerankingTask with {len(self.datasets)} datasets")
        print(f"  Datasets: {self.datasets}")

    def downsample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downsample reranking queries."""
        downsample = self.config.get("downsample")
        if downsample is None or downsample >= 1.0:
            return df

        n_sample = max(1, int(len(df) * downsample))
        df_sampled = df.sample(n=n_sample, random_state=self.seed_general).reset_index(drop=True)

        print(f"Downsampled to {n_sample} queries ({downsample*100:.1f}%)")
        return df_sampled

    def load_hf_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load the specified reranking dataset.

        Args:
            dataset_name: Name of the dataset to load (normalized)

        Returns:
            DataFrame containing the reranking dataset with columns:
            - query: The search query
            - positive: List of relevant documents
            - negative: List of irrelevant documents

        Raises:
            ValueError: If dataset_name is not supported
            Exception: If dataset loading fails
        """
        # Get HuggingFace dataset name from metadata (handles alternate names)
        metadata = self.dataset_metadata[self.get_canonical_name(dataset_name)]
        hf_dataset_name = metadata.hf_dataset_name

        try:
            from datasets import load_dataset

            print(f"Loading reranking dataset: {dataset_name} ({hf_dataset_name})")

            # Load dataset from HuggingFace Hub
            dataset = load_dataset(hf_dataset_name, split=self.split)

            # Convert to DataFrame
            df = dataset.to_pandas()

            # Check if this is a multi-config format (corpus/queries/qrels/top_ranked)
            # Identified by having query-id, corpus-id, score columns (qrels format)
            if (
                "query-id" in df.columns
                and "corpus-id" in df.columns
                and "score" in df.columns
            ):
                print(
                    "  Detected multi-config format (corpus/queries/qrels/top_ranked)"
                )
                df = self._load_multi_config_format(hf_dataset_name)
            else:
                # Standard format - validate and normalize
                required_columns = ["query", "positive", "negative"]
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]

                if missing_columns:
                    # Try alternative column names or data structure
                    df = self._normalize_dataset_columns(df, dataset_name)

            print(f"✓ Loaded {len(df)} samples from {dataset_name}")

            # Log dataset statistics
            self._log_dataset_stats(df, dataset_name)

            # Apply downsampling if configured
            df = self.downsample_data(df)

            return df

        except Exception as e:
            error_msg = f"Failed to load reranking dataset '{dataset_name}': {str(e)}"
            print(f"ERROR: {error_msg}")
            raise Exception(error_msg) from e

    def _load_multi_config_format(self, hf_dataset_name: str) -> pd.DataFrame:
        """
        Load reranking dataset from multi-config format (corpus/queries/qrels/top_ranked).

        This format is used by datasets like RuBQReranking where data is split across
        multiple configs:
        - corpus: Documents with _id, text, title
        - queries: Queries with _id, text
        - default: Relevance judgments (qrels) with query-id, corpus-id, score
        - top_ranked: Pre-ranked lists with query-id, corpus-ids (list)

        Args:
            hf_dataset_name: Full HuggingFace dataset name (e.g., 'mteb/RuBQReranking')

        Returns:
            DataFrame with columns: query, positive, negative
        """
        from datasets import load_dataset

        print(f"  Loading multi-config format for {hf_dataset_name}")

        # Load all required configs
        corpus = load_dataset(hf_dataset_name, name="corpus", split=self.split)
        queries = load_dataset(hf_dataset_name, name="queries", split=self.split)
        qrels = load_dataset(hf_dataset_name, name="default", split=self.split)
        top_ranked = load_dataset(hf_dataset_name, name="top_ranked", split=self.split)

        print(
            f"    Loaded: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels, {len(top_ranked)} rankings"
        )

        # Create lookup dictionaries for efficient access
        corpus_dict = {row["_id"]: row["text"] for row in corpus}
        queries_dict = {row["_id"]: row["text"] for row in queries}

        # Build qrels lookup: {query_id: {corpus_id: score}}
        qrels_dict = {}
        for row in qrels:
            qid = row["query-id"]
            if qid not in qrels_dict:
                qrels_dict[qid] = {}
            qrels_dict[qid][row["corpus-id"]] = row["score"]

        # Construct DataFrame with query, positive, negative format
        rows = []
        for row in top_ranked:
            qid = row["query-id"]
            query_text = queries_dict.get(qid, "")

            if not query_text:
                print(f"    Warning: Query {qid} not found in queries config, skipping")
                continue

            corpus_ids = row["corpus-ids"]

            # Separate positive and negative based on qrels scores
            positive_docs = []
            negative_docs = []

            for cid in corpus_ids:
                doc_text = corpus_dict.get(cid, "")
                if not doc_text:
                    print(
                        f"    Warning: Document {cid} not found in corpus config, skipping"
                    )
                    continue

                # Get relevance score (default to 0 if not in qrels)
                score = qrels_dict.get(qid, {}).get(cid, 0)

                if score > 0:
                    positive_docs.append(doc_text)
                else:
                    negative_docs.append(doc_text)

            # Only add if we have at least one positive and one negative document
            if positive_docs and negative_docs:
                rows.append(
                    {
                        "query": query_text,
                        "positive": positive_docs,
                        "negative": negative_docs,
                    }
                )

        df = pd.DataFrame(rows)
        print(f"    Constructed {len(df)} reranking samples from multi-config format")

        return df

    def _normalize_dataset_columns(
        self, df: pd.DataFrame, dataset_name: str
    ) -> pd.DataFrame:
        """
        Normalize dataset columns to standard format.

        Args:
            df: Raw dataset DataFrame
            dataset_name: Name of the dataset

        Returns:
            DataFrame with normalized columns (query, positive, negative)
        """
        # Handle different column naming conventions
        column_mappings = {
            # Common alternative column names
            "question": "query",
            "text": "query",
            "sentence": "query",
            "query_text": "query",
            "pos": "positive",
            "positives": "positive",
            "relevant": "positive",
            "pos_docs": "positive",
            "neg": "negative",
            "negatives": "negative",
            "irrelevant": "negative",
            "neg_docs": "negative",
        }

        # Apply column mappings
        df = df.rename(columns=column_mappings)

        # Ensure lists for positive and negative - handle pandas arrays safely
        def safe_list_conversion(x):
            """Safely convert various array types to Python lists."""
            try:
                if x is None:
                    return []
                # Handle strings specially - don't iterate over characters
                if isinstance(x, str):
                    return [x]
                # Handle pandas arrays, numpy arrays, and other iterables
                if hasattr(x, "__iter__"):
                    return list(x)
                # Handle single values
                return [x]
            except (TypeError, ValueError):
                return []

        for col in ["positive", "negative"]:
            if col in df.columns:
                df[col] = df[col].apply(safe_list_conversion)

        # Validate that we now have required columns
        required_columns = ["query", "positive", "negative"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Dataset {dataset_name} is missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

        return df

    def _log_dataset_stats(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Log statistics about the loaded dataset.

        Args:
            df: The loaded dataset DataFrame
            dataset_name: Name of the dataset
        """
        try:
            # Count documents - handle both pandas arrays and Python lists
            def safe_len(x):
                """Safely get length of list-like object, handling pandas arrays."""
                try:
                    if hasattr(x, "__len__"):
                        return (
                            len(list(x))
                            if hasattr(x, "__iter__") and not isinstance(x, str)
                            else len(x)
                        )
                    return 0
                except (TypeError, AttributeError):
                    return 0

            def is_empty(x):
                """Safely check if list-like object is empty, handling pandas arrays."""
                try:
                    if x is None:
                        return True
                    # Convert to list if it's array-like but not string
                    if hasattr(x, "__iter__") and not isinstance(x, str):
                        x_list = list(x)
                        return len(x_list) == 0
                    return len(x) == 0
                except (TypeError, AttributeError, ValueError):
                    return True

            total_positive = sum(safe_len(pos) for pos in df["positive"])
            total_negative = sum(safe_len(neg) for neg in df["negative"])

            avg_pos = total_positive / len(df) if len(df) > 0 else 0
            avg_neg = total_negative / len(df) if len(df) > 0 else 0

            print("  Dataset statistics:")
            print(f"    Total queries: {len(df)}")
            print(
                f"    Total positive documents: {total_positive} (avg: {avg_pos:.1f} per query)"
            )
            print(
                f"    Total negative documents: {total_negative} (avg: {avg_neg:.1f} per query)"
            )

            # Check for empty samples - use safe checking
            empty_pos = sum(1 for pos in df["positive"] if is_empty(pos))
            empty_neg = sum(1 for neg in df["negative"] if is_empty(neg))

            if empty_pos > 0:
                print(f"    Warning: {empty_pos} queries have no positive documents")
            if empty_neg > 0:
                print(f"    Warning: {empty_neg} queries have no negative documents")

        except Exception as e:
            print(f"    Could not compute dataset statistics: {e}")

    def evaluate(
        self,
        dataset_name: str,
        embedding_model: SentenceTransformer,
        emb_model_name: str,
        data: Optional[pd.DataFrame] = None,
        gen_model_name: Optional[str] = None,
        gen_api_name: Optional[str] = None,
        pregenerated_paraphrases: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate the embedding model on the reranking dataset with paraphrases.

        Args:
            dataset_name: Name of the dataset
            embedding_model: The embedding model to evaluate
            data: Optional pre-loaded dataset
            gen_model_name: Name of generative model for paraphrases
            gen_api_name: API name for generative model
            pregenerated_paraphrases: Pre-generated paraphrases to use
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation results with original and paraphrase comparisons
        """
        if data is None:
            data = self.load_hf_data(dataset_name)

        # Convert DataFrame to samples format expected by evaluator
        samples = self._convert_dataframe_to_samples(data)

        # Handle paraphrases if generative model is specified
        if gen_model_name and gen_api_name:
            if pregenerated_paraphrases:
                # Use pregenerated paraphrases
                print(
                    f"Processing with pre-generated paraphrases for {gen_model_name}..."
                )

                # Reconstruct paraphrased samples from parent class format
                paraphrased_samples = self._reconstruct_paraphrased_samples(
                    original_samples=samples, paraphrases=pregenerated_paraphrases
                )

                # Evaluate both original and paraphrased
                original_result = self._evaluate_samples(
                    samples, dataset_name, embedding_model, **kwargs
                )
                paraphrased_result = self._evaluate_samples(
                    paraphrased_samples, dataset_name, embedding_model, **kwargs
                )

                # Combine results with comparison
                result = self._combine_reranking_results(
                    original_result,
                    paraphrased_result,
                    emb_model_name,
                    gen_model_name,
                    gen_api_name,
                )

                return result

            else:
                # Generate paraphrases on-demand (online mode)
                print(f"Generating paraphrases on-demand for {gen_model_name}...")

                paraphrased_samples = self.generate_paraphrases(
                    samples=samples,
                    gen_model=gen_model_name,
                    gen_api=gen_api_name,
                    seed=self.seed_paraphrasing,
                    dataset_name=dataset_name,
                )

                # Evaluate both original and paraphrased
                original_result = self._evaluate_samples(
                    samples, dataset_name, embedding_model, **kwargs
                )
                paraphrased_result = self._evaluate_samples(
                    paraphrased_samples, dataset_name, embedding_model, **kwargs
                )

                # Combine results with comparison
                result = self._combine_reranking_results(
                    original_result,
                    paraphrased_result,
                    emb_model_name,
                    gen_model_name,
                    gen_api_name,
                )

                return result
        else:
            # No paraphrases - evaluate original data only
            result = self._evaluate_samples(
                samples, dataset_name, embedding_model, **kwargs
            )
            result.update(
                {
                    "emb_model": emb_model_name,
                    "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
                    "model_max_seq_length": kwargs.get(
                        "model_max_seq_length",
                        getattr(embedding_model, "max_seq_length", None),
                    ),
                    "gen_model": None,
                    "gen_api": None,
                    "emb_model_api": self.config.get(
                        "emb_model_api", "sentence-transformers"
                    ),
                    "emb_batch_size": self.config.get("emb_batch_size", 32),
                    "dtype": str(self.config.get("st_dtype", "torch.float32")),
                    "encoding_chunk_size": self.config.get("encoding_chunk_size", None),
                    # Missing fields for consistency
                    "timestamp": self.timestamp,
                    "run": kwargs.get("run", 0),
                    "temperature": None,
                    "top_p": None,
                    "embedding_generation_time": kwargs.get(
                        "embedding_generation_time", None
                    ),
                }
            )
            return result

    def _convert_dataframe_to_samples(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert DataFrame to samples format expected by evaluator.

        Args:
            data: DataFrame with query, positive, negative columns

        Returns:
            List of sample dictionaries
        """

        def safe_to_list(x):
            """Safely convert array-like object to Python list."""
            try:
                if x is None:
                    return []
                # Handle pandas arrays, numpy arrays, and other iterables
                if hasattr(x, "__iter__") and not isinstance(x, str):
                    return list(x)
                # Handle single values (convert to list)
                return [x] if x else []
            except (TypeError, ValueError):
                return []

        samples = []

        for _, row in data.iterrows():
            # Safely convert pandas arrays to Python lists
            positive_docs = safe_to_list(row["positive"])
            negative_docs = safe_to_list(row["negative"])

            sample = {
                "query": row["query"],
                "positive": positive_docs,
                "negative": negative_docs,
            }

            # Only include samples with both positive and negative documents
            if len(sample["positive"]) > 0 and len(sample["negative"]) > 0:
                samples.append(sample)

        print(f"Converted {len(samples)} valid samples from DataFrame")

        # Validate that we have at least some valid samples
        if len(samples) == 0:
            total_samples = len(data)
            raise ValueError(
                f"No valid samples found (samples must have both positive and negative documents). "
                f"Original dataset had {total_samples} samples. "
                f"Please check data quality."
            )

        return samples

    def _evaluate_samples(
        self,
        samples: List[Dict[str, Any]],
        dataset_name: str,
        embedding_model: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate samples using the reranking evaluator.

        Args:
            samples: List of reranking samples
            dataset_name: Name of the dataset
            embedding_model: The embedding model to evaluate
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation results
        """
        # Create encode_kwargs with batch_size from config
        encode_kwargs = {
            "batch_size": self.config.get("emb_batch_size", 128),
        }

        # Create evaluator with samples
        evaluator = RerankingEvaluator(
            samples=samples,
            # k_values=self.k_values,
            use_batched_encoding=self.use_batched_encoding,
            seed=self.seed_general,
            encode_kwargs=encode_kwargs,
        )

        try:
            # Run evaluation (this follows the MTEB standard)
            results = evaluator(model=embedding_model, task_name=dataset_name, **kwargs)
            return results
        except Exception as e:
            error_msg = f"Evaluation failed for {dataset_name}: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {"error": error_msg, "map": 0.0, "mrr": 0.0}

    def _reconstruct_paraphrased_samples(
        self, original_samples: List[Dict[str, Any]], paraphrases: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Reconstruct paraphrased samples from parent class paraphrase format.

        Args:
            original_samples: Original samples
            paraphrases: Paraphrases in parent class format (with paraphrases1)

        Returns:
            List of paraphrased samples maintaining reranking structure
        """
        if "paraphrases1" not in paraphrases:
            print(
                "Warning: No paraphrases1 found in pregenerated data, returning original samples"
            )
            return original_samples

        paraphrased_texts = paraphrases["paraphrases1"]

        # Reconstruct samples with paraphrased texts
        text_idx = 0
        paraphrased_samples = []

        for sample in original_samples:
            num_pos = len(sample["positive"])
            num_neg = len(sample["negative"])
            total = 1 + num_pos + num_neg

            query = paraphrased_texts[text_idx]
            positives = paraphrased_texts[text_idx + 1 : text_idx + 1 + num_pos]
            negatives = paraphrased_texts[text_idx + 1 + num_pos : text_idx + total]

            paraphrased_samples.append(
                {
                    "query": query,
                    "positive": positives,
                    "negative": negatives,
                }
            )

            text_idx += total

        return paraphrased_samples

    def _combine_reranking_results(
        self,
        original_result: Dict[str, Any],
        paraphrased_result: Dict[str, Any],
        emb_model_name: str,
        gen_model_name: str,
        gen_api_name: str,
    ) -> Dict[str, Any]:
        """
        Combine original and paraphrased evaluation results.

        Args:
            original_result: Results from original samples
            paraphrased_result: Results from paraphrased samples
            emb_model_name: Embedding model name
            gen_model_name: Generative model name
            gen_api_name: Generative model API name

        Returns:
            Combined results dictionary
        """
        # Extract key metrics for comparison
        original_map = original_result.get("map", 0.0)
        paraphrased_map = paraphrased_result.get("map", 0.0)
        original_mrr = original_result.get("mrr", 0.0)
        paraphrased_mrr = paraphrased_result.get("mrr", 0.0)

        # Calculate differences
        map_diff = paraphrased_map - original_map
        mrr_diff = paraphrased_mrr - original_mrr

        # Combine results with comparison metrics
        result = {
            "emb_model": emb_model_name,
            "gen_model": gen_model_name,
            "gen_api": gen_api_name,
            "emb_model_api": self.config.get("emb_model_api", "sentence-transformers"),
            "emb_batch_size": self.config.get("emb_batch_size", 32),
            "dtype": str(self.config.get("st_dtype", "torch.float32")),
            # Missing fields for consistency
            "timestamp": self.timestamp,
            "run": 0,  # Default run for paraphrased results
            "temperature": self.temperature,
            "top_p": self.top_p,
            # Original results (with prefix)
            "original_map": original_map,
            "original_mrr": original_mrr,
            "original_main_score": original_map,
            # Paraphrased results (with prefix)
            "paraphrased_map": paraphrased_map,
            "paraphrased_mrr": paraphrased_mrr,
            "paraphrased_main_score": paraphrased_map,
            # Differences
            "map_diff": map_diff,
            "mrr_diff": mrr_diff,
            # Primary metrics (use paraphrased as main)
            "map": paraphrased_map,
            "mrr": paraphrased_mrr,
        }

        # Add any additional metrics from both results
        for key, value in original_result.items():
            if key not in result and not key.startswith("original_"):
                result[f"original_{key}"] = value

        for key, value in paraphrased_result.items():
            if key not in result and not key.startswith("paraphrased_"):
                result[f"paraphrased_{key}"] = value
                if key not in result:  # Also include without prefix for main metrics
                    result[key] = value

        return result

    def _evaluate_online(
        self,
        samples: List[Dict[str, Any]],
        dataset_name: str,
        embedding_model: Any,
        evaluator: RerankingEvaluator,
        emb_model_name: str | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate in online mode (standard MTEB benchmark).

        Args:
            samples: List of reranking samples
            dataset_name: Name of the dataset
            embedding_model: The embedding model
            evaluator: The reranking evaluator
            emb_model_name: Name of the embedding model (optional, will fallback to getattr)
            **kwargs: Additional parameters

        Returns:
            Dictionary of evaluation results
        """
        # Use provided model name or fallback to getattr
        if emb_model_name is None:
            emb_model_name = getattr(embedding_model, "model_name", "unknown_model")

        print("=== Online Reranking Evaluation (MTEB benchmark) ===")

        # Standard evaluation
        results = evaluator(model=embedding_model, task_name=dataset_name, **kwargs)

        # Add model information
        result = {"embedding_model": emb_model_name, **results}

        return result

    def generate_paraphrases(
        self,
        samples: List[Dict[str, Any]],
        gen_model: str,
        gen_api: str,
        seed: int = 1337,
        dataset_name: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate paraphrases for reranking samples (queries + positive/negative documents).

        Args:
            samples: List of reranking samples with query, positive, negative
            gen_model: Generative model name
            gen_api: API name for the generative model
            seed: Random seed for reproducibility
            dataset_name: Name of the dataset (for consistent saving)

        Returns:
            List of samples with paraphrased queries and documents
        """
        # Extract all texts to paraphrase: queries + all positive docs + all negative docs
        all_texts = []
        text_mapping = []  # Track which sample and type each text belongs to

        for sample_idx, sample in enumerate(samples):
            # Add query
            all_texts.append(sample["query"])
            text_mapping.append(
                {"sample_idx": sample_idx, "type": "query", "doc_idx": 0}
            )

            # Add positive documents
            for doc_idx, doc in enumerate(sample["positive"]):
                all_texts.append(doc)
                text_mapping.append(
                    {"sample_idx": sample_idx, "type": "positive", "doc_idx": doc_idx}
                )

            # Add negative documents
            for doc_idx, doc in enumerate(sample["negative"]):
                all_texts.append(doc)
                text_mapping.append(
                    {"sample_idx": sample_idx, "type": "negative", "doc_idx": doc_idx}
                )

        # Use single-run approach for Type 1 architecture
        paraphrased_texts = self._generate_paraphrases(
            texts=all_texts,
            gen_model_name=gen_model,
            gen_api_name=gen_api,
            dataset_name=dataset_name,
            seed=seed,
        )

        # Reconstruct paraphrased samples maintaining the reranking structure
        paraphrased_samples = []
        for sample in samples:
            paraphrased_samples.append({"query": "", "positive": [], "negative": []})

        # Map paraphrased texts back to samples
        for i, (paraphrased_text, mapping) in enumerate(
            zip(paraphrased_texts, text_mapping)
        ):
            sample_idx = mapping["sample_idx"]
            text_type = mapping["type"]
            doc_idx = mapping["doc_idx"]

            if text_type == "query":
                paraphrased_samples[sample_idx]["query"] = paraphrased_text
            elif text_type == "positive":
                # Ensure the positive list is long enough
                while len(paraphrased_samples[sample_idx]["positive"]) <= doc_idx:
                    paraphrased_samples[sample_idx]["positive"].append("")
                paraphrased_samples[sample_idx]["positive"][doc_idx] = paraphrased_text
            elif text_type == "negative":
                # Ensure the negative list is long enough
                while len(paraphrased_samples[sample_idx]["negative"]) <= doc_idx:
                    paraphrased_samples[sample_idx]["negative"].append("")
                paraphrased_samples[sample_idx]["negative"][doc_idx] = paraphrased_text

        # Save paraphrases if requested
        if self.save_paraphrases:
            seeds_used = [seed] * len(samples)
            self._save_paraphrases(
                original_samples=samples,
                paraphrased_samples=paraphrased_samples,
                gen_model_name=gen_model,
                gen_api_name=gen_api,
                dataset_name=dataset_name,
                seeds_used=seeds_used,
            )

        return paraphrased_samples

    def _save_paraphrases(
        self,
        original_samples: List[Dict[str, Any]],
        paraphrased_samples: List[Dict[str, Any]],
        gen_model_name: str,
        gen_api_name: str,
        dataset_name: str,
        seeds_used: Optional[List[Optional[int]]] = None,
        split_override: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ) -> None:
        """
        Save generated paraphrases to disk with reranking-specific format.

        Args:
            original_samples: Original samples with query, positive, negative structure
            paraphrased_samples: Paraphrased samples with same structure
            gen_model_name: Name of the generative model
            gen_api_name: API name for the generative model
            dataset_name: Name of the dataset being processed
            seeds_used: Optional list of seeds used for each paraphrase
            split_override: Override the split name used in the file path
            languages: Optional list of language codes for each sample
        """

        from ..utils import save_df

        # Create list of records for DataFrame
        records = []
        for i, (orig_sample, para_sample) in enumerate(
            zip(original_samples, paraphrased_samples)
        ):
            record = {
                "original_query": orig_sample["query"],
                "paraphrase_query": para_sample["query"],
                "original_positive": orig_sample["positive"],
                "paraphrase_positive": para_sample["positive"],
                "original_negative": orig_sample["negative"],
                "paraphrase_negative": para_sample["negative"],
            }

            records.append(record)

        # Create DataFrame
        paraphrase_df = pd.DataFrame(records)

        # Add language information
        if languages:
            paraphrase_df["language"] = languages
        else:
            # Get language from dataset metadata
            metadata = self.dataset_metadata.get(self.get_canonical_name(dataset_name))
            if metadata and metadata.languages:
                # Use first language from metadata as default
                default_lang = (
                    metadata.languages[0]
                    if isinstance(metadata.languages, list)
                    else metadata.languages
                )
            else:
                # Fallback to English
                default_lang = "eng"
            paraphrase_df["language"] = [default_lang] * len(paraphrase_df)

        # Determine output directory
        effective_split = split_override if split_override is not None else self.split

        # Use canonical name for path, add multilingual suffix if needed (only for STS17/STS22)
        dataset_name_for_path = self.get_canonical_name(dataset_name)
        if should_use_multilingual_suffix(dataset_name_for_path, self.config):
            dataset_name_for_path = f"{dataset_name_for_path}_multilingual"

        para_output_dir = os.path.join(
            "data",
            "paraphrases",
            "reranking",
            dataset_name_for_path,
            effective_split,
            re.sub(r":", "-", gen_model_name),
        )

        os.makedirs(para_output_dir, exist_ok=True)

        # Create filename with new seed-run pattern
        if should_use_multilingual_suffix(dataset_name, self.config):
            para_filename = f"{self.timestamp}_{dataset_name}_{gen_model_name}_paraphrases_multilingual"
        else:
            para_filename = (
                f"{self.timestamp}_{dataset_name}_{gen_model_name}_paraphrases"
            )

        # Add seed and run number - use provided seeds_used or default to task seed
        current_seed = (
            seeds_used[0]
            if seeds_used and len(seeds_used) > 0
            else self.seed_paraphrasing
        )
        if self.n_runs > 1:
            # Determine run number from seed
            if current_seed != self.seed_paraphrasing:
                run_offset = self.get_run_index_for_seed(current_seed)
                run_number = run_offset + 1
            else:
                run_number = 1
            para_filename += f"_seed-{current_seed:04d}_run-{run_number}"
        else:
            # Single run case
            para_filename += f"_seed-{current_seed:04d}_run-1"
        para_filename = re.sub(
            r":", "-", para_filename
        )  # Sanitize model name in filename

        # Use JSONL format to preserve list structures
        output_file = os.path.join(para_output_dir, para_filename)

        # Save using the utility function
        save_df(paraphrase_df, output_file, file_format="jsonl")

        print(f"Saved reranking paraphrases to {output_file}")

    def run_full_evaluation(
        self, embedding_models: List[Any], emb_model_names: List[str], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run complete evaluation across all datasets and embedding models.

        Args:
            embedding_models: List of embedding models to evaluate
            emb_model_names: List of model name strings corresponding to embedding_models
            **kwargs: Additional evaluation parameters

        Returns:
            List of all evaluation results
        """
        all_results = []

        for dataset_name in self.datasets:
            print(f"\n{'=' * 50}")
            print(f"Evaluating dataset: {dataset_name}")
            print(f"{'=' * 50}")

            # Load dataset once for all evaluations
            try:
                data = self.load_hf_data(dataset_name)
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {e}")
                continue

            # Pre-generate paraphrases once for this dataset
            paraphrases_cache = {}

            for model_idx, embedding_model in enumerate(embedding_models):
                emb_model_name = emb_model_names[model_idx]
                print(f"\nEvaluating with embedding model: {emb_model_name}")

                if not self.gen_model:
                    # Evaluate without paraphrases
                    print("  Evaluating original data only (no generative models)")

                    try:
                        result = self.evaluate(
                            dataset_name=dataset_name,
                            embedding_model=embedding_model,
                            emb_model_name=emb_model_name,
                            data=data,
                            **kwargs,
                        )

                        # Add metadata to result
                        result.update(
                            {
                                "dataset": self.get_canonical_name(dataset_name),
                                "n_samples": len(data),
                                "timestamp": self.timestamp,
                                "task": self.task_name,
                                "runtime": 0.0,  # Will be updated by caller if needed
                                "seed_general": self.seed_general,
                                "seed_paraphrasing": self.seed_paraphrasing,
                                "gen_model": None,
                                "gen_api": None,
                                "paraphrase_generation_time": None,
                            }
                        )

                        all_results.append(result)

                        # Print immediate results
                        if "map" in result:
                            map_score = result["map"]
                            # mrr_score = result.get("mrr", 0)
                            print(
                                f"   MAP: {map_score:.4f}"
                            )  # , MRR@{self.mrr_at_k}: {mrr_score:.4f}")

                        # Display main score (map)
                        if "map" in result:
                            print(f"  Main Score (MAP): {result['map']:.4f}")

                    except Exception as e:
                        print(f"    Error in reranking evaluation: {e}")

                        # Add error result
                        error_result = {
                            "embedding_model": emb_model_name,
                            "dataset": self.get_canonical_name(dataset_name),
                            "n_samples": len(data) if "data" in locals() else 0,
                            "timestamp": self.timestamp,
                            "task": self.task_name,
                            "runtime": 0.0,
                            "seed_general": self.seed_general,
                            "seed_paraphrasing": self.seed_paraphrasing,
                            "gen_model": None,
                            "gen_api": None,
                            "paraphrase_generation_time": None,
                            "error": str(e),
                            "map": 0.0,
                            "mrr": 0.0,
                        }
                        all_results.append(error_result)
                        continue

                else:
                    # Evaluate with paraphrases using Type 1 architecture (run loop)
                    gen_api = self.gen_model_api

                    # Add run loop for Type 1 architecture
                    for run in range(self.n_runs):
                        run_seed = self.get_seed_for_run(run)
                        print(
                            f"  With paraphrases from: {self.gen_model} ({gen_api}) - Run {run + 1}/{self.n_runs} (seed: {run_seed})"
                        )

                        try:
                            # Generate paraphrases for this specific run
                            original_samples = self._convert_dataframe_to_samples(data)

                            para_start = time.time()
                            paraphrased_samples = self.generate_paraphrases(
                                samples=original_samples,
                                gen_model=self.gen_model,
                                gen_api=gen_api,
                                seed=run_seed,
                                dataset_name=dataset_name,
                            )
                            para_time = time.time() - para_start

                            # Evaluate both original and paraphrased samples directly
                            original_result = self._evaluate_samples(
                                original_samples,
                                dataset_name,
                                embedding_model,
                                **kwargs,
                            )
                            paraphrased_result = self._evaluate_samples(
                                paraphrased_samples,
                                dataset_name,
                                embedding_model,
                                **kwargs,
                            )

                            # Combine results with comparison
                            result = self._combine_reranking_results(
                                original_result,
                                paraphrased_result,
                                emb_model_name,
                                self.gen_model,
                                gen_api,
                            )

                            # Add metadata to result
                            result.update(
                                {
                                    "dataset": self.get_canonical_name(dataset_name),
                                    "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
                                    "model_max_seq_length": kwargs.get(
                                        "model_max_seq_length",
                                        getattr(
                                            embedding_model, "max_seq_length", None
                                        ),
                                    ),
                                    "encoding_chunk_size": self.config.get(
                                        "encoding_chunk_size", None
                                    ),
                                    "n_samples": len(original_samples),
                                    "timestamp": self.timestamp,
                                    "task": self.task_name,
                                    "runtime": 0.0,  # Will be updated by caller if needed
                                    "seed_general": self.seed_general,
                                    "seed_paraphrasing": run_seed,  # Use run-specific seed (already incremented)
                                    "run": run + 1,  # Add run information
                                    "gen_model": self.gen_model,
                                    "gen_api": gen_api,
                                    "paraphrase_generation_time": para_time,
                                }
                            )

                            all_results.append(result)

                            # Print immediate results
                            if "original_map" in result and "paraphrased_map" in result:
                                orig_map = result["original_map"]
                                para_map = result["paraphrased_map"]
                                orig_mrr = result.get("original_mrr", 0)
                                para_mrr = result.get("paraphrased_mrr", 0)
                                # print(
                                #     f"    Original MAP: {orig_map:.4f}, MRR@{self.mrr_at_k}: {orig_mrr:.4f}"
                                # )
                                # print(
                                #     f"    Paraphrased MAP: {para_map:.4f}, MRR@{self.mrr_at_k}: {para_mrr:.4f}"
                                # )

                                # Display main scores
                                if (
                                    "original_main_score" in result
                                    and "paraphrased_main_score" in result
                                ):
                                    print(
                                        f"    Main Score - Original: {result['original_main_score']:.4f}, Paraphrased: {result['paraphrased_main_score']:.4f}"
                                    )

                            elif "map" in result:
                                map_score = result["map"]
                                mrr_score = result.get("mrr", 0)
                                # print(
                                #     f"    MAP: {map_score:.4f}, MRR@{self.mrr_at_k}: {mrr_score:.4f}"
                                # )

                                # Display main score
                                if "main_score" in result:
                                    print(f"    Main Score: {result['main_score']:.4f}")

                        except Exception as e:
                            print(
                                f"    Error in reranking evaluation with {self.gen_model} run {run + 1}: {e}"
                            )

                            # Add error result
                            error_result = {
                                "embedding_model": emb_model_name,
                                "dataset": self.get_canonical_name(dataset_name),
                                "n_samples": (
                                    len(paraphrases_cache["original_samples"])
                                    if paraphrases_cache
                                    else 0
                                ),
                                "timestamp": self.timestamp,
                                "task": self.task_name,
                                "runtime": 0.0,
                                "seed_general": self.seed_general,
                                "seed_paraphrasing": self.seed_paraphrasing,
                                "gen_model": self.gen_model,
                                "gen_api": gen_api,
                                "paraphrase_generation_time": para_time,
                                "error": str(e),
                                "map": 0.0,
                                "mrr": 0.0,
                                "main_score": 0.0,
                            }
                            all_results.append(error_result)
                            continue

        return all_results

    def __str__(self) -> str:
        """String representation of the task."""
        return f"RerankingTask(datasets={self.datasets}"  # , mrr_at_k={self.mrr_at_k})"
