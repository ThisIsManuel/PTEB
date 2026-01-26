"""
Clustering task implementation for PTEB.

This module implements the ClusteringTask class that handles clustering
dataset loading, paraphrase generation/loading, and evaluation with clustering metrics.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..evaluators.clustering_evaluator import ClusteringEvaluator
from ..utils import (
    cleanup_model,
    should_use_multilingual_suffix,
)
from .paraphrase_task import ParaphraseTask


class ClusteringTask(ParaphraseTask):
    """
    Clustering task with paraphrase-enhanced evaluation.

    This class handles:
    - Loading clustering datasets from HuggingFace Hub
    - Generating or loading paraphrases
    - Evaluating embedding models on original and paraphrased data
    - Computing clustering metrics (v-measure, homogeneity, completeness)
    """

    def __init__(self, datasets: List[str], config: Dict[str, Any], **kwargs):
        """
        Initialize the Clustering task.

        Args:
            datasets: List of clustering dataset names to evaluate on
            config: Configuration dictionary from CLI arguments
            **kwargs: Additional parameters
        """
        super().__init__(
            task_name="Clustering",
            task_description="Clustering evaluation with paraphrase augmentation",
            datasets=datasets,
            config=config,
            **kwargs,
        )

        # Initialize Clustering evaluator with task-specific parameters
        # MTEB default is 500 for clustering_batch_size (Mini-Batch K-Means batch size)
        clustering_batch_size = config.get("clustering_batch_size", 500)

        self.evaluator = ClusteringEvaluator(
            clustering_batch_size=clustering_batch_size,
            task_name=self.task_name,
            seed=self.seed_general,
        )

    def downsample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downsample clustering groups."""
        downsample = self.config.get("downsample")
        if downsample is None or downsample >= 1.0:
            return df

        n_sample = max(1, int(len(df) * downsample))
        df_sampled = df.sample(n=n_sample, random_state=self.seed_general).reset_index(drop=True)

        print(f"Downsampled to {n_sample} clustering groups ({downsample*100:.1f}%)")
        return df_sampled

    def _standardize_col_names(
        self, df: pd.DataFrame, dataset_name: str
    ) -> pd.DataFrame:
        """
        Standardize column names for clustering datasets.

        Args:
            df: DataFrame with original dataset columns
            dataset_name: Name of the dataset being processed (used for error messages)

        Returns:
            DataFrame with standardized column names ('sentences' and 'labels')
        """

        # Standardize column names for clustering tasks
        # Clustering datasets typically have 'sentences' and 'labels' columns
        if "text" in df.columns and "sentences" not in df.columns:
            df["sentences"] = df["text"]
        elif "sentence" in df.columns and "sentences" not in df.columns:
            df["sentences"] = df["sentence"]
        elif "title" in df.columns and "sentences" not in df.columns:
            # Some clustering datasets use 'title' as the main text column
            df["sentences"] = df["title"]

        # Handle different label column names
        if "category" in df.columns and "labels" not in df.columns:
            df["labels"] = df["category"]
        elif "label" in df.columns and "labels" not in df.columns:
            df["labels"] = df["label"]
        elif "cluster" in df.columns and "labels" not in df.columns:
            df["labels"] = df["cluster"]

        # Verify required columns exist
        required_columns = ["sentences", "labels"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            available_columns = list(df.columns)
            raise ValueError(
                f"Dataset {dataset_name} is missing required columns: {missing_columns}. "
                f"Available columns: {available_columns}"
            )

        return df

    def load_hf_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load and return the specified clustering dataset from HuggingFace Hub.

        Uses dataset metadata to determine the correct HuggingFace dataset path
        and loading strategy (monolingual vs multilingual).

        Args:
            dataset_name: Name of the clustering dataset to load

        Returns:
            DataFrame containing the loaded dataset with sentences and labels columns

        Raises:
            ValueError: If dataset_name is not supported
            FileNotFoundError: If dataset cannot be loaded
        """
        # Get dataset metadata
        metadata = self.dataset_metadata[self.get_canonical_name(dataset_name)]

        try:
            print(
                f"Loading dataset {metadata.hf_dataset_name} (split: {self.split}) from HuggingFace Hub..."
            )

            from datasets import load_dataset

            # Determine which languages to load based on metadata and config
            languages = self.config.get("languages", None)

            if metadata.is_non_en():
                # Multilingual dataset - load with language configs
                # Use config languages if specified, otherwise use all from metadata
                if languages and len(languages) > 0:
                    lang_configs = [lang.lower() for lang in languages]
                else:
                    lang_configs = [lang.lower() for lang in metadata.languages]

                print(
                    f"  Loading {len(lang_configs)} language config(s): {lang_configs}"
                )

                all_dfs = []
                for config in lang_configs:
                    ds = load_dataset(
                        metadata.hf_dataset_name, config, split=self.split
                    )
                    lang_df = pd.DataFrame(ds)
                    lang_df = self._standardize_col_names(lang_df, dataset_name)
                    lang_df["language"] = config
                    all_dfs.append(lang_df)

                if not all_dfs:
                    raise ValueError(
                        f"Failed to load any language configs for {dataset_name}. "
                        f"Tried configs: {lang_configs}"
                    )

                df = pd.concat(all_dfs, ignore_index=True)
                print(
                    f"✓ Loaded {len(all_dfs)} language config(s) from HuggingFace Hub"
                )
                df = self.downsample_data(df)
                return df
            else:
                # Monolingual dataset - load without config
                hf_dataset = load_dataset(metadata.hf_dataset_name, split=self.split)
                df = pd.DataFrame(hf_dataset)
                df = self._standardize_col_names(df, dataset_name)
                print("✓ Loaded monolingual dataset from HuggingFace Hub")
                df = self.downsample_data(df)
                return df

        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load dataset {metadata.hf_dataset_name}: {str(e)}"
            )

    def load_data(self, dataset_name: str) -> Dict[str, pd.DataFrame]:
        """
        Load dataset from HuggingFace Hub.
        Returns raw ungrouped data (grouping by language applied later in run loop).

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Dictionary with 'original' DataFrame (ungrouped)
        """
        # Load from HuggingFace Hub
        original_df = self.load_hf_data(dataset_name)
        data = {"original": original_df}

        return data

    def evaluate_model(
        self,
        embedding_model: Any,
        dataset_name: str,
        data: pd.DataFrame,
        paraphrased_data: pd.DataFrame | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluate embedding model on clustering data with nested structure.
        Each row represents a separate clustering problem to be evaluated independently.

        Args:
            embedding_model: The embedding model to evaluate
            dataset_name: Name of the dataset being evaluated
            data: Original dataset with nested structure (each row has lists of sentences/labels)
            paraphrased_data: Optional paraphrased dataset with same structure

        Returns:
            Dictionary containing aggregated evaluation results across all groups
        """
        results = {}

        # Evaluate original data - each row is a separate clustering problem
        print(
            f"\nEvaluating {dataset_name} on original data ({len(data)} clustering groups)..."
        )
        original_v_measures = []
        language_scores = {}  # Track scores per language if available
        language_stats = {}  # Track detailed stats per language (n_examples, n_labels)

        for group_idx, row in data.iterrows():
            sentences = row["sentences"]
            labels = row["labels"]

            # Convert labels to strings for consistency
            labels = [str(label) for label in labels]

            # Get language if available
            language = row.get("language", None)
            lang_str = f" ({language})" if language else ""

            # Evaluate this group
            group_results = self.evaluator(
                embedding_model, sentences=sentences, labels=labels
            )
            v_measure = group_results["v_measure"]
            original_v_measures.append(v_measure)

            # Track scores and stats by language
            if language:
                if language not in language_scores:
                    language_scores[language] = []
                    language_stats[language] = {"n_examples": 0, "n_labels": set()}
                language_scores[language].append(v_measure)
                language_stats[language]["n_examples"] += len(sentences)
                language_stats[language]["n_labels"].update(labels)

            print(
                f"  Group {group_idx + 1}/{len(data)}{lang_str}: {len(sentences)} sentences, "
                f"{len(set(labels))} unique labels → v_measure: {v_measure:.4f}"
            )

        # Calculate statistics across all groups
        original_stats = {
            "v_measure": float(np.mean(original_v_measures)),
            "v_measure_std": float(np.std(original_v_measures)),
            "v_measure_min": float(np.min(original_v_measures)),
            "v_measure_max": float(np.max(original_v_measures)),
            "n_groups": len(original_v_measures),
            "individual_scores": original_v_measures,
        }
        results["original"] = original_stats

        print(
            f"  ✓ Original results: mean={original_stats['v_measure']:.4f}, std={original_stats['v_measure_std']:.4f}"
        )

        # Evaluate on paraphrased data if available
        if paraphrased_data is not None and not paraphrased_data.empty:
            print(
                f"\nEvaluating {dataset_name} on paraphrased data ({len(paraphrased_data)} clustering groups)..."
            )

            if len(paraphrased_data) != len(data):
                raise ValueError(
                    f"Paraphrased data has {len(paraphrased_data)} groups but original has {len(data)}"
                )

            paraphrased_v_measures = []
            para_language_scores = {}  # Track scores per language if available
            para_language_stats = {}  # Track detailed stats per language

            for group_idx, row in paraphrased_data.iterrows():
                sentences = row["paraphrased_sentences"]
                labels = row["labels"]

                # Convert labels to strings for consistency
                labels = [str(label) for label in labels]

                # Get language if available
                language = row.get("language", None)
                lang_str = f" ({language})" if language else ""

                # Evaluate this group
                group_results = self.evaluator(
                    embedding_model, sentences=sentences, labels=labels
                )
                v_measure = group_results["v_measure"]
                paraphrased_v_measures.append(v_measure)

                # Track scores and stats by language
                if language:
                    if language not in para_language_scores:
                        para_language_scores[language] = []
                        para_language_stats[language] = {
                            "n_examples": 0,
                            "n_labels": set(),
                        }
                    para_language_scores[language].append(v_measure)
                    para_language_stats[language]["n_examples"] += len(sentences)
                    para_language_stats[language]["n_labels"].update(labels)

                print(
                    f"  Group {group_idx + 1}/{len(paraphrased_data)}{lang_str}: {len(sentences)} sentences, "
                    f"{len(set(labels))} unique labels → v_measure: {v_measure:.4f}"
                )

            # Calculate statistics across all groups
            paraphrased_stats = {
                "v_measure": float(np.mean(paraphrased_v_measures)),
                "v_measure_std": float(np.std(paraphrased_v_measures)),
                "v_measure_min": float(np.min(paraphrased_v_measures)),
                "v_measure_max": float(np.max(paraphrased_v_measures)),
                "n_groups": len(paraphrased_v_measures),
                "individual_scores": paraphrased_v_measures,
            }
            results["paraphrased"] = paraphrased_stats

            print(
                f"  ✓ Paraphrased results: mean={paraphrased_stats['v_measure']:.4f}, std={paraphrased_stats['v_measure_std']:.4f}"
            )

            # Print per-language results table if available
            if language_stats and para_language_stats:
                print("\n  Per-language results:")
                print(
                    f"  {'Language':<12} {'#Examples':<12} {'#Labels':<10} "
                    f"{'Original':<12} {'Paraphrase':<12}"
                )
                print("  " + "-" * 68)
                for lang in sorted(language_stats.keys()):
                    n_examples = language_stats[lang]["n_examples"]
                    n_labels = len(language_stats[lang]["n_labels"])
                    orig_score = np.mean(language_scores[lang])
                    para_score = (
                        np.mean(para_language_scores[lang])
                        if lang in para_language_scores
                        else 0.0
                    )
                    print(
                        f"  {lang:<12} {n_examples:<12} {n_labels:<10} "
                        f"{orig_score:<12.4f} {para_score:<12.4f}"
                    )
            elif language_stats:
                # Original only (no paraphrases)
                print("\n  Per-language results:")
                print(
                    f"  {'Language':<12} {'#Examples':<12} {'#Labels':<10} {'Original':<12}"
                )
                print("  " + "-" * 46)
                for lang in sorted(language_stats.keys()):
                    n_examples = language_stats[lang]["n_examples"]
                    n_labels = len(language_stats[lang]["n_labels"])
                    orig_score = np.mean(language_scores[lang])
                    print(
                        f"  {lang:<12} {n_examples:<12} {n_labels:<10} {orig_score:<12.4f}"
                    )

            # Compute performance difference (using mean values)
            v_measure_diff = (
                paraphrased_stats["v_measure"] - original_stats["v_measure"]
            )
            results["performance_difference"] = {
                "v_measure_diff": v_measure_diff,
                "v_measure_relative_change": (
                    (v_measure_diff / original_stats["v_measure"]) * 100
                    if original_stats["v_measure"] != 0
                    else 0
                ),
            }
        else:
            # No paraphrases - print language table for original data only
            if language_stats:
                print("\n  Per-language results:")
                print(
                    f"  {'Language':<12} {'#Examples':<12} {'#Labels':<10} {'Original':<12}"
                )
                print("  " + "-" * 46)
                for lang in sorted(language_stats.keys()):
                    n_examples = language_stats[lang]["n_examples"]
                    n_labels = len(language_stats[lang]["n_labels"])
                    orig_score = np.mean(language_scores[lang])
                    print(
                        f"  {lang:<12} {n_examples:<12} {n_labels:<10} {orig_score:<12.4f}"
                    )

        return results

    def evaluate(
        self,
        dataset_name: str,
        embedding_model: Any,
        emb_model_name: str,
        data: pd.DataFrame | None = None,
        gen_model_name: str | None = None,
        gen_api_name: str | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate the embedding model on clustering data with paraphrases.

        Args:
            dataset_name: Name of the dataset being evaluated
            embedding_model: The embedding model to evaluate
            data: DataFrame with the dataset (if None, will load from dataset_name)
            gen_model_name: Name of generative model for paraphrases
            gen_api_name: API name for generative model
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation results and metadata
        """
        start_time = time.time()

        # Load data if not provided
        if data is None:
            data = self.load_hf_data(dataset_name)

        # Count total individual sentences across all clustering groups
        # Since we now assume nested structure, each row contains a list of sentences
        total_sentences = sum(len(row["sentences"]) for _, row in data.iterrows())
        result = {
            "timestamp": self.timestamp,
            "run": kwargs.get("run", 0),
            "dataset": self.get_canonical_name(dataset_name),
            "emb_model": emb_model_name,
            "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
            "model_max_seq_length": kwargs.get(
                "model_max_seq_length", getattr(embedding_model, "max_seq_length", None)
            ),
            "emb_model_api": self.config.get("emb_model_api", "sentence-transformers"),
            "emb_batch_size": self.config.get("emb_batch_size"),
            "dtype": str(self.config.get("st_dtype")),
            "encoding_chunk_size": self.config.get("encoding_chunk_size", None),
            "gen_model": gen_model_name,
            "n_samples": total_sentences,
            "task": self.task_name,
            "seed_general": self.seed_general,
            "seed_paraphrasing": self.seed_paraphrasing,
        }

        paraphrased_data = None
        paraphrase_generation_time = None

        # Handle paraphrases if generative model is specified
        if gen_model_name and gen_api_name:
            # Generate paraphrases on the fly
            para_start_time = time.time()
            paraphrased_data = self._generate_paraphrases_nested(
                data, gen_model_name, gen_api_name, dataset_name
            )
            paraphrase_generation_time = time.time() - para_start_time
            result["paraphrase_generation_time"] = paraphrase_generation_time

        # Evaluate model
        embedding_start_time = time.time()
        evaluation_results = self.evaluate_model(
            embedding_model, dataset_name, data, paraphrased_data
        )
        embedding_generation_time = time.time() - embedding_start_time
        result["embedding_generation_time"] = embedding_generation_time

        # Flatten evaluation results to individual columns (instead of nested dictionaries)
        self._flatten_clustering_results(evaluation_results, result)

        # Calculate total time
        total_time = time.time() - start_time
        result["total_evaluation_time"] = total_time

        # Add temperature and top_p fields
        if gen_model_name:
            result["temperature"] = self.temperature
            result["top_p"] = self.top_p
        else:
            result["temperature"] = None
            result["top_p"] = None

        return result

    def _flatten_clustering_results(
        self, evaluation_results: Dict[str, Any], main_result: Dict[str, Any]
    ) -> None:
        """
        Flatten clustering evaluation results from nested dictionary to individual columns.
        Now handles per-group statistics (mean, std, min, max) across clustering groups.

        Args:
            evaluation_results: Nested results from evaluate_model with per-group statistics
            main_result: Main result dictionary to update
        """
        # Get original results (now contains aggregated statistics across groups)
        original_results = evaluation_results.get("original", {})

        # Add original metrics - main v-measure is the mean across groups
        main_result["original_v_measure"] = original_results.get("v_measure", 0.0)
        main_result["original_v_measure_std"] = original_results.get(
            "v_measure_std", 0.0
        )
        main_result["original_v_measure_min"] = original_results.get(
            "v_measure_min", 0.0
        )
        main_result["original_v_measure_max"] = original_results.get(
            "v_measure_max", 0.0
        )
        main_result["n_clustering_groups"] = original_results.get("n_groups", 0)

        # Set main score to mean v_measure for original data
        main_result["original_main_score"] = original_results.get("v_measure", 0.0)

        # Add paraphrased metrics if available
        paraphrased_results = evaluation_results.get("paraphrased", {})
        if paraphrased_results:
            main_result["paraphrased_v_measure"] = paraphrased_results.get(
                "v_measure", 0.0
            )
            main_result["paraphrased_v_measure_std"] = paraphrased_results.get(
                "v_measure_std", 0.0
            )
            main_result["paraphrased_v_measure_min"] = paraphrased_results.get(
                "v_measure_min", 0.0
            )
            main_result["paraphrased_v_measure_max"] = paraphrased_results.get(
                "v_measure_max", 0.0
            )
            main_result["paraphrased_main_score"] = paraphrased_results.get(
                "v_measure", 0.0
            )

            # Add performance difference metrics (using mean values)
            performance_diff = evaluation_results.get("performance_difference", {})
            main_result["v_measure_diff"] = performance_diff.get("v_measure_diff", 0.0)
            main_result["v_measure_relative_change"] = performance_diff.get(
                "v_measure_relative_change", 0.0
            )
        else:
            # When no paraphrased results are available, set paraphrased metrics to 0.0
            main_result["paraphrased_v_measure"] = 0.0
            main_result["paraphrased_v_measure_std"] = 0.0
            main_result["paraphrased_v_measure_min"] = 0.0
            main_result["paraphrased_v_measure_max"] = 0.0
            main_result["paraphrased_main_score"] = 0.0
            main_result["v_measure_diff"] = 0.0
            main_result["v_measure_relative_change"] = 0.0

    def run_full_evaluation(
        self, embedding_models: List[Any], model_names: List[str], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run complete evaluation across all datasets, embedding models, and generative models.

        Args:
            embedding_models: List of embedding models to evaluate
            model_names: List of model name strings corresponding to embedding_models
            **kwargs: Additional evaluation parameters

        Returns:
            List of all evaluation results
        """
        all_results = []

        for dataset_name in self.datasets:
            print(f"\n{'=' * 50}")
            print(f"Evaluating dataset: {dataset_name}")
            print(f"{'=' * 50}")

            # Load dataset - returns ungrouped data with seed_paraphrasing preserved
            try:
                loaded_data = self.load_data(dataset_name)
                data_original = loaded_data["original"]
                data_paraphrased = loaded_data.get("paraphrased", None)
            except Exception as e:
                # If loading fails for this dataset, skip to next dataset
                print(f"Error loading dataset {dataset_name}: {e}")
                continue

            # No cache needed - paraphrases are generated on-the-fly
            paraphrase_time = None

            for model_idx, embedding_model in enumerate(embedding_models):
                emb_model_name = model_names[model_idx]
                print(f"\nEvaluating with embedding model: {emb_model_name}")

                if not self.gen_model:
                    # Evaluate without paraphrases
                    print("  Evaluating original data only (no generative models)")

                    try:
                        result = self.evaluate(
                            dataset_name=dataset_name,
                            embedding_model=embedding_model,
                            emb_model_name=emb_model_name,
                            data=data_original,
                        )

                        # Add metadata (don't override n_samples - it's already set correctly by evaluate())
                        result.update(
                            {
                                "dataset": self.get_canonical_name(dataset_name),
                                "timestamp": self.timestamp,
                                "task": self.task_name,
                                "runtime": 0.0,
                                "seed_general": self.seed_general,
                                "seed_paraphrasing": self.seed_paraphrasing,
                                "gen_model": None,
                                "gen_api": None,
                                "paraphrase_generation_time": None,
                            }
                        )
                        all_results.append(result)

                        # Print results (using flattened structure)
                        if "original_v_measure" in result:
                            v_measure = result["original_v_measure"]
                            print(f"    V-measure: {v_measure:.4f}")

                        # Display main score (v_measure)
                        if "original_main_score" in result:
                            print(
                                f"  Main Score (v_measure): {result['original_main_score']:.4f}"
                            )

                    except Exception as e:
                        print(f"    Error in evaluation: {e}")
                        continue
                else:
                    # Evaluate with paraphrases
                    for run in range(self.n_runs):
                        run_seed = self.get_seed_for_run(run)
                        print(
                            f"  With paraphrases from: {self.gen_model} ({self.gen_model_api}) - Run {run + 1}/{self.n_runs} (seed: {run_seed})"
                        )

                        try:
                            # Use consistent variable naming
                            original_data_run = data_original

                            # Generate paraphrases for this specific run
                            para_start = time.time()
                            paraphrased_data_run = self._generate_paraphrases_nested(
                                original_data_run,
                                self.gen_model,
                                self.gen_model_api,
                                dataset_name,
                                seed=run_seed,
                            )
                            para_time = time.time() - para_start

                            # Evaluate using generated data
                            evaluation_results = self.evaluate_model(
                                embedding_model=embedding_model,
                                dataset_name=dataset_name,
                                data=original_data_run,
                                paraphrased_data=paraphrased_data_run,
                            )

                            # Create base result dictionary with metadata
                            result = {
                                "dataset": self.get_canonical_name(dataset_name),
                                "emb_model": emb_model_name,
                                "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
                                "model_max_seq_length": kwargs.get(
                                    "model_max_seq_length",
                                    getattr(embedding_model, "max_seq_length", None),
                                ),
                                "emb_model_api": self.config.get(
                                    "emb_model_api", "sentence-transformers"
                                ),
                                "emb_batch_size": self.config.get("emb_batch_size", 32),
                                "dtype": str(
                                    self.config.get("st_dtype", "torch.float32")
                                ),
                                "encoding_chunk_size": self.config.get(
                                    "encoding_chunk_size", None
                                ),
                                "gen_model": self.gen_model,
                                "n_samples": len(original_data_run),
                                "task": self.task_name,
                                "seed_general": self.seed_general,
                                "seed_paraphrasing": run_seed,  # Use run-specific seed
                                "run": run + 1,  # Add run information
                                "timestamp": self.timestamp,
                                "runtime": 0.0,
                                "gen_api": self.gen_model_api,
                                "paraphrase_generation_time": para_time,
                                "embedding_generation_time": 0.0,  # Will be updated if needed
                            }

                            # Flatten the nested evaluation results
                            self._flatten_clustering_results(evaluation_results, result)
                            all_results.append(result)

                            # Print results (using flattened structure)
                            if "original_v_measure" in result:
                                orig_v = result["original_v_measure"]
                                print(f"      Original V-measure: {orig_v:.4f}")
                            if "paraphrased_v_measure" in result:
                                para_v = result["paraphrased_v_measure"]
                                print(f"      Paraphrased V-measure: {para_v:.4f}")

                            # Display main scores
                            if (
                                "original_main_score" in result
                                and "paraphrased_main_score" in result
                            ):
                                print(
                                    f"    Main Score - Original: {result['original_main_score']:.4f}, Paraphrased: {result['paraphrased_main_score']:.4f}"
                                )

                        except Exception as e:
                            print(f"    Error with run {run + 1}: {e}")
                            continue

                # Clean up embedding model after evaluation to free GPU memory
                cleanup_model(embedding_model, emb_model_name)

        return all_results

    def _generate_paraphrases_nested(
        self,
        data: pd.DataFrame,
        gen_model_name: str,
        gen_api_name: str,
        dataset_name: str,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate paraphrases for clustering data with nested structure.

        This method preserves clustering's unique nested DataFrame structure where:
        - Each row represents a clustering group (monolingual) or language group (multilingual)
        - Each row contains lists of sentences and labels

        Args:
            data: Original clustering dataset with nested structure (each row has lists of sentences/labels)
            gen_model_name: Name of the generative model
            gen_api_name: API name for the generative model
            dataset_name: Name of the dataset being processed
            seed: Optional seed for this specific run

        Returns:
            DataFrame with same nested structure but paraphrased sentences
        """
        # Determine the seed to use for this run
        actual_seed = seed if seed is not None else self.seed_paraphrasing

        paraphrased_rows = []

        for group_idx, row in data.iterrows():
            sentences = row["sentences"]
            labels = row["labels"]

            print(
                f"  Generating paraphrases for group {group_idx + 1}/{len(data)}: {len(sentences)} sentences"
            )

            # Call parent's method to generate paraphrases with explicit seed
            paraphrased_sentences = super()._generate_paraphrases(
                texts=sentences,
                gen_model_name=gen_model_name,
                gen_api_name=gen_api_name,
                dataset_name=dataset_name,
                seed=actual_seed,
            )

            # Create row with paraphrased sentences and original labels (maintain nested format)
            paraphrased_row = {
                "paraphrased_sentences": paraphrased_sentences,
                "labels": labels,
            }

            # Preserve language column if it exists (needed for multilingual datasets)
            if "language" in row:
                paraphrased_row["language"] = row["language"]

            paraphrased_rows.append(paraphrased_row)

        # Create DataFrame with same nested structure
        paraphrased_data = pd.DataFrame(paraphrased_rows)

        # Save paraphrases if requested (maintain nested structure)
        if self.save_paraphrases:
            seeds_used = [actual_seed] * len(paraphrased_data)
            self._save_paraphrases(
                paraphrased_data=paraphrased_data,
                original_data=data,
                gen_model_name=gen_model_name,
                gen_api_name=gen_api_name,
                dataset_name=dataset_name,
                seeds_used=seeds_used,
            )

        return paraphrased_data

    def _save_paraphrases(
        self,
        paraphrased_data: pd.DataFrame,
        original_data: pd.DataFrame,
        gen_model_name: str,
        gen_api_name: str,
        dataset_name: str,
        seeds_used: Optional[List[Optional[int]]] = None,
        split_override: Optional[str] = None,
    ) -> None:
        """
        Save generated paraphrases to disk with clustering-specific nested format.

        Args:
            paraphrased_data: DataFrame with nested paraphrased structure
            original_data: DataFrame with nested original structure
            gen_model_name: Name of the generative model
            gen_api_name: API name for the generative model
            dataset_name: Name of the dataset being processed
            seeds_used: Optional list of seeds used for each paraphrase
            split_override: Override the split name used in the file path
        """

        from ..utils import save_df

        # Create list to store all group records
        all_records = []

        # Process each group to create records with nested structure
        for para_row, orig_row in zip(
            paraphrased_data.iterrows(), original_data.iterrows()
        ):
            para_row_data = para_row[1]
            orig_row_data = orig_row[1]

            # Create record with nested structure
            record = {
                "original_sentences": orig_row_data["sentences"],
                "paraphrased_sentences": para_row_data["paraphrased_sentences"],
                "labels": orig_row_data["labels"],
            }

            # Preserve language column if it exists (needed for multilingual datasets)
            if "language" in orig_row_data:
                record["language"] = orig_row_data["language"]
            else:
                # Get default language from metadata if not in data
                metadata = self.dataset_metadata.get(
                    self.get_canonical_name(dataset_name)
                )
                if metadata and metadata.languages:
                    # Use first language from metadata as default
                    default_lang = (
                        metadata.languages[0]
                        if isinstance(metadata.languages, list)
                        else metadata.languages
                    )
                    record["language"] = default_lang
                else:
                    # Fallback to English
                    record["language"] = "eng"

            all_records.append(record)

        # Create DataFrame from records
        paraphrase_df = pd.DataFrame(all_records)

        # The run information is encoded in the filename instead

        # Determine output directory
        effective_split = split_override if split_override is not None else self.split

        # Use canonical name for path, add multilingual suffix if needed (only for STS17/STS22)
        dataset_name_for_path = self.get_canonical_name(dataset_name)
        if should_use_multilingual_suffix(dataset_name_for_path, self.config):
            dataset_name_for_path = f"{dataset_name_for_path}_multilingual"

        para_output_dir = os.path.join(
            "data",
            "paraphrases",
            "clustering",
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

        run_offset = self.get_run_index_for_seed(current_seed)
        run_number = run_offset + 1
        para_filename += f"_seed-{current_seed:04d}_run-{run_number}"

        para_filename = re.sub(
            r":", "-", para_filename
        )  # Sanitize model name in filename

        # Use JSONL format for consistency
        output_file = os.path.join(para_output_dir, para_filename)

        # Save using the utility function
        save_df(paraphrase_df, output_file, file_format="jsonl")

        print(f"Saved clustering paraphrases to {output_file}")
