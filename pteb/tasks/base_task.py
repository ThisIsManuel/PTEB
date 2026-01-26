"""
Abstract base class for all evaluation tasks in PTEB.

This module defines the BaseTask class that follows MTEB's proven patterns
for task abstraction and provides a consistent interface for all evaluation tasks.
"""

import datetime
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sentence_transformers import SentenceTransformer

from ..utils import (
    apply_language_filter,
    load_all_files,
    normalize_punctuation_spacing,
)


class BaseTask(ABC):
    """
    Abstract base class for all evaluation tasks.

    Following MTEB's design patterns, this class defines the interface
    that all evaluation tasks must implement while providing common
    functionality for task metadata, timing, and result management.
    """

    def __init__(
        self,
        task_name: str,
        task_description: str,
        datasets: list[str],
        config: dict[str, Any],
        **kwargs,
    ):
        """
        Initialize the base task with metadata and configuration.

        Args:
            task_name: Name of the task (e.g., "STS", "Classification")
            task_description: Human-readable description of the task
            datasets: List of dataset names this task can evaluate on
            config: Configuration dictionary from CLI arguments
            **kwargs: Additional task-specific parameters
        """
        self.task_name = task_name
        self.task_description = task_description
        self.datasets = datasets
        self.config = config

        # Extract common configuration
        seed_general = config.get("seed_general", config.get("seed"))
        seed_paraphrasing = config.get("seed_paraphrasing", config.get("seed"))

        if seed_general is None:
            seed_general = random.randint(0, 2**32 - 1)
        if seed_paraphrasing is None:
            seed_paraphrasing = random.randint(0, 2**32 - 1)

        self.seed_general = seed_general
        self.seed_paraphrasing = seed_paraphrasing

        # Split can be optional in config - will use dataset metadata if not specified
        self.split = config.get("split", None)

        # Initialize timing and result tracking
        self.start_time: float | None = None
        self.results: list[dict[str, Any]] = []
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

        # Load dataset metadata for each dataset
        self.dataset_metadata: dict[str, Any] = {}
        self._input_to_canonical: dict[str, str] = {}
        self._load_dataset_metadata()

        # If split not specified in config, use dataset metadata's eval_split
        # Use the first dataset's eval_split as default for the whole task
        if self.split is None:
            if self.datasets:
                first_canonical = self.get_canonical_name(self.datasets[0])
                if first_canonical in self.dataset_metadata:
                    first_dataset_metadata = self.dataset_metadata[first_canonical]
                    if hasattr(first_dataset_metadata, "eval_split"):
                        self.split = first_dataset_metadata.eval_split
                        print(
                            f"Using eval_split '{self.split}' from {first_canonical} metadata"
                        )
                    else:
                        self.split = "test"  # Fallback
                else:
                    self.split = "test"  # Fallback if no metadata
            else:
                self.split = "test"  # Fallback if no datasets

        # Validate configuration
        self._validate_config()

    @classmethod
    def get_main_score_name(cls, task_name: str) -> str:
        """
        Get the name of the main score metric for this evaluator.

        Returns:
            Name of the main score metric
        """

        # Import evaluators dynamically to avoid circular imports
        from ..evaluators import (
            classification_evaluator,
            clustering_evaluator,
            pair_classification_evaluator,
            reranking_evaluator,
            retrieval_evaluator,
            sts_evaluator,
            summarization_evaluator,
        )

        # Map task names to their evaluators and get the main score name
        task_name_lower = task_name.lower()

        if task_name_lower == "classification":
            return (
                classification_evaluator.ClassificationEvaluator.get_main_score_name()
            )
        elif task_name_lower == "clustering":
            return clustering_evaluator.ClusteringEvaluator.get_main_score_name()
        elif task_name_lower == "pair_classification":
            return pair_classification_evaluator.PairClassificationEvaluator.get_main_score_name()
        elif task_name_lower == "reranking":
            return reranking_evaluator.RerankingEvaluator.get_main_score_name()
        elif task_name_lower == "retrieval":
            return retrieval_evaluator.RetrievalEvaluator.get_main_score_name()
        elif task_name_lower == "sts":
            return sts_evaluator.STSEvaluator.get_main_score_name()
        elif task_name_lower == "summarization":
            return summarization_evaluator.SummarizationEvaluator.get_main_score_name()
        else:
            raise ValueError(f"Unsupported task type: {task_name}")

    def get_seed_for_run(self, run_index: int) -> int:
        """
        Get the seed for a specific run index.

        - Integer: increments from base seed (seed + run_index)

        Args:
            run_index: The run index (0-based)

        Returns:
            The seed value for this specific run

        """
        return self.seed_paraphrasing + run_index

    def get_run_index_for_seed(self, seed: int) -> int:
        """
        Get the run index for a specific seed value (inverse of get_seed_for_run).

        Args:
            seed: The seed value

        Returns:
            The run index (0-based) for this seed
        """
        return seed - self.seed_paraphrasing

    def _load_dataset_metadata(self) -> None:
        """
        Load metadata for each dataset in self.datasets.

        This method attempts to load metadata for each dataset from the
        dataset_metadata module. If a dataset doesn't have registered metadata,
        it creates a fallback default metadata object.

        Metadata is stored keyed by canonical name (metadata.name), and a
        mapping from input names to canonical names is maintained.
        """
        from ..dataset_metadata import get_dataset_metadata

        for dataset_name in self.datasets:
            try:
                metadata = get_dataset_metadata(dataset_name)
                canonical_name = metadata.name
                self.dataset_metadata[canonical_name] = metadata
                self._input_to_canonical[dataset_name] = canonical_name
                print(
                    f"[OK] Loaded metadata for {dataset_name} -> {canonical_name}: {metadata.hf_dataset_name}"
                )
            except ValueError as e:
                # Dataset not registered - create fallback metadata
                print(f"Warning: {e}")
                print(f"  Creating default metadata for {dataset_name}")
                self.dataset_metadata[dataset_name] = self._create_default_metadata(
                    dataset_name
                )
                self._input_to_canonical[dataset_name] = dataset_name

    def get_canonical_name(self, dataset_name: str) -> str:
        """
        Get the canonical dataset name from any input name.

        Args:
            dataset_name: Input dataset name (can be alternate or canonical)

        Returns:
            Canonical dataset name from metadata, or input name if not found
        """
        return self._input_to_canonical.get(dataset_name, dataset_name)

    def _create_default_metadata(self, dataset_name: str) -> Any:
        """
        Create a default metadata object for datasets without registered metadata.

        This is a fallback mechanism to maintain backward compatibility with
        datasets that haven't been added to the dataset_metadata module yet.

        Args:
            dataset_name: Name of the dataset

        Returns:
            A simple metadata object with default values
        """
        from ..dataset_metadata import BaseDatasetMetadata

        # Create a dynamic class with default values
        class DefaultMetadata(BaseDatasetMetadata):
            name = dataset_name
            hf_dataset_name = f"mteb/{dataset_name}"
            default_splits = ["test"]
            languages = None  # Assume monolingual English
            is_multilingual = False
            task_type = self.task_name.lower()
            description = f"Auto-generated metadata for {dataset_name}"
            eval_split = "test"  # Default evaluation split
            train_split = None  # No training split by default

        return DefaultMetadata()

    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        if not self.datasets:
            raise ValueError("At least one dataset must be specified")

    def _load_paraphrase_files(
        self, directory: str | os.PathLike, file_type: str = "jsonl"
    ) -> pd.DataFrame:
        """
        Load and concatenate all paraphrase files from a directory.

        This is a helper method that wraps the load_all_files utility function
        to provide consistent paraphrase file loading across all task implementations.
        If language columns exist, applies language filtering based on config.
        If no language columns exist, uses all data without filtering.

        Args:
            directory: Path to the directory containing paraphrase files
            file_type: Type of files to load (default: "jsonl")

        Returns:
            DataFrame with concatenated data from all files, with optional language filtering

        Raises:
            FileNotFoundError: If directory doesn't exist or no files of the specified type found
            ValueError: If no valid files could be loaded
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Paraphrase directory not found: {directory}")

        df = load_all_files(directory, file_type)

        # Normalize punctuation spacing in paraphrase columns
        # This fixes extra spaces around punctuation (e.g., " : " → ":", " ( " → "(")
        # and strips leading/trailing whitespace and newlines
        paraphrase_cols = [col for col in df.columns if "paraphrase" in col.lower()]
        for col in paraphrase_cols:
            df[col] = df[col].apply(normalize_punctuation_spacing)

        # Also strip original text columns (sentence1, sentence2, text, query, etc.)
        original_text_cols = [
            col
            for col in df.columns
            if col
            in [
                "sentence1",
                "sentence2",
                "text",
                "query",
                "document",
                "positive",
                "negative",
            ]
            or col.startswith("original_")
        ]
        for col in original_text_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: x.strip() if isinstance(x, str) else x
                )

        # Only apply language filtering if language column exists in the data
        # If no language column, use all rows (no filtering needed)
        if "language" in df.columns or "lang" in df.columns:
            df = apply_language_filter(df, self.config)
        else:
            print(
                "Info: No language column found in paraphrase files, using all data without language filtering"
            )

        return df

    def map_run_to_available_seed(
        self, seed_ids: list[int], requested_seed: int, run_number: int, n_runs: int
    ) -> int:
        """
        Map a run number to an available seed from the data.

        When the requested seed is not available in the data, this method will
        use the Nth lowest available seed (where N = run_number). This ensures
        that each run uses a unique seed even if the requested seeds don't match
        the available seeds in the paraphrase files.

        Args:
            seed_ids: List of all seed IDs from the paraphrase data
            requested_seed: The seed that was originally requested
            run_number: The current run number (1-indexed)
            n_runs: Total number of runs configured

        Returns:
            The actual seed to use for this run

        Raises:
            ValueError: If there aren't enough unique seeds for the number of runs
        """
        # Get all available seeds and sort them
        available_seeds = sorted(set(seed_ids))

        # Check if we have enough seeds for the requested number of runs
        if len(available_seeds) < n_runs:
            raise ValueError(
                f"Not enough unique seeds in paraphrase data. "
                f"Available seeds: {available_seeds} (n={len(available_seeds)}), "
                f"but config requires n_runs={n_runs}."
            )

        # Map run number to the appropriate seed from available seeds
        # Run 1 uses available_seeds[0], Run 2 uses available_seeds[1], etc.
        actual_seed = available_seeds[run_number - 1]

        if actual_seed != requested_seed:
            print(f"Note: Requested seed {requested_seed} not available in data.")
            print(f"      Available seeds: {available_seeds}")
            print(f"      Using seed {actual_seed} for run {run_number}")

        return actual_seed

    @abstractmethod
    def load_hf_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load and return the specified dataset.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            DataFrame containing the loaded dataset

        Raises:
            ValueError: If dataset_name is not supported
            FileNotFoundError: If dataset files are not found
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        dataset_name: str,
        embedding_model: SentenceTransformer,
        emb_model_name: str,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate the embedding model on the specified dataset.

        Args:
            dataset_name: Name of the dataset to evaluate on
            embedding_model: The embedding model to evaluate
            emb_model_name: Name of the embedding model (optional)
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation results
        """
        pass

    def save_results(self, output_path: str, file_format: str = "xlsx") -> None:
        """
        Save evaluation results to file.

        Args:
            output_path: Base path for output file (without extension)
            file_format: Output format ("xlsx", "csv", "json", "jsonl")
        """
        if not self.results:
            print("No results to save")
            return

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create results DataFrame
        results_df = pd.DataFrame(self.results)

        # Save based on format
        if file_format == "xlsx":
            results_df.to_excel(f"{output_path}.xlsx", index=False)
        elif file_format == "csv":
            results_df.to_csv(f"{output_path}.csv", index=False)
        elif file_format == "json":
            results_df.to_json(f"{output_path}.json", orient="records", indent=2)
        elif file_format == "jsonl":
            results_df.to_json(f"{output_path}.jsonl", orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        print(f"Results saved to {output_path}.{file_format}")

    def run_evaluation(
        self, embedding_models: list[Any], **kwargs
    ) -> list[dict[str, Any]]:
        """
        Run evaluation across all datasets and embedding models.

        Args:
            embedding_models: List of embedding models to evaluate
            **kwargs: Additional evaluation parameters

        Returns:
            List of evaluation results
        """
        self.start_time = time.time()

        print(f"\n{'#' * 60}")
        print(f"Starting {self.task_name} evaluation")
        print(f"{'#' * 60}")

        all_results = []

        for dataset_name in self.datasets:
            print(f"\n{'=' * 40}")
            print(f"Dataset: {dataset_name}")
            print(f"{'=' * 40}")

            try:
                # Load dataset
                data = self.load_hf_data(dataset_name)

                # Evaluate with each embedding model
                for emb_model in embedding_models:
                    print(
                        f"\nEvaluating with model: {getattr(emb_model, 'model_name', 'unknown_model')}"
                    )

                    eval_start = time.time()

                    # Run evaluation
                    result = self.evaluate(
                        dataset_name=dataset_name,
                        embedding_model=emb_model,
                        data=data,
                        **kwargs,
                    )

                    # Add metadata to result
                    result.update(
                        {
                            "timestamp": self.timestamp,
                            "task": self.task_name,
                            "dataset": self.get_canonical_name(dataset_name),
                            "n_samples": len(data),
                            "runtime": time.time() - eval_start,
                            "seed_general": self.seed_general,
                            "seed_paraphrasing": self.seed_paraphrasing,
                        }
                    )

                    self.results.append(result)
                    all_results.append(result)

                    print(f"Completed in {result['runtime']:.2f} seconds")

            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
                continue

        total_time = time.time() - self.start_time
        print(f"\nTotal evaluation time: {total_time:.2f} seconds")

        return all_results

    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary table of evaluation results.

        Returns:
            DataFrame with key metrics for each evaluation
        """
        if not self.results:
            return pd.DataFrame()

        return pd.DataFrame(self.results)

    def __str__(self) -> str:
        """String representation of the task."""
        return f"{self.task_name}Task(datasets={self.datasets})"

    def __repr__(self) -> str:
        """Detailed string representation of the task."""
        return (
            f"{self.__class__.__name__}("
            f"task_name='{self.task_name}', "
            f"datasets={self.datasets})"
        )
