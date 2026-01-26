"""
Pair Classification task implementation for PTEB.

This module implements the PairClassificationTask class that handles pair classification
dataset loading, paraphrase generation/loading, and evaluation with classification metrics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..evaluators.pair_classification_evaluator import PairClassificationEvaluator
from ..utils import (
    cleanup_model,
    should_use_multilingual_suffix,
)
from .paraphrase_task import ParaphraseTask

logger = logging.getLogger(__name__)


class PairClassificationTask(ParaphraseTask):
    """
    Pair Classification task with paraphrase-enhanced evaluation.

    This class handles:
    - Loading pair classification datasets from HuggingFace Hub
    - Generating or loading paraphrases
    - Evaluating embedding models on original and paraphrased data
    - Computing classification metrics (accuracy, F1, precision, recall, AP)
    """

    def __init__(self, datasets: List[str], config: Dict[str, Any], **kwargs):
        """
        Initialize the Pair Classification task.

        Args:
            datasets: List of pair classification dataset names to evaluate on
            config: Configuration dictionary from CLI arguments
            **kwargs: Additional parameters
        """
        super().__init__(
            task_name="PairClassification",
            task_description="Pair Classification evaluation with paraphrase augmentation",
            datasets=datasets,
            config=config,
            **kwargs,
        )

        # Duplicate removal configuration
        self.remove_duplicates = config.get("remove_duplicates", True)

        # Initialize Pair Classification evaluator
        self.evaluator = PairClassificationEvaluator(seed=self.seed_general)

    def downsample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downsample sentence pairs DataFrame."""
        downsample = self.config.get("downsample")
        if downsample is None or downsample >= 1.0:
            return df

        n_sample = max(1, int(len(df) * downsample))
        df_sampled = df.sample(n=n_sample, random_state=self.seed_general)

        print(f"Downsampled to {n_sample} pairs ({downsample*100:.1f}%)")
        return df_sampled

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize column names to handle different naming conventions
        # Handle sentence columns
        if "sent1" in df.columns and "sent2" in df.columns:
            # Rename from sent1/sent2 to sentence1/sentence2
            df = df.rename(columns={"sent1": "sentence1", "sent2": "sentence2"})
        elif not ("sentence1" in df.columns and "sentence2" in df.columns):
            # Check for other common variations
            cols = df.columns.tolist()
            raise ValueError(
                f"Missing required sentence columns. Expected 'sentence1'/'sentence2' or 'sent1'/'sent2'. Available: {cols}"
            )

        # Handle label columns
        if "labels" in df.columns:
            # Rename from labels to label
            df = df.rename(columns={"labels": "label"})
        elif "label" not in df.columns:
            raise ValueError(
                f"Missing label column. Expected 'label' or 'labels'. Available: {list(df.columns)}"
            )

        return df

    def load_hf_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load and return the specified pair classification dataset from HuggingFace Hub.

        Args:
            dataset_name: Name of the pair classification dataset to load

        Returns:
            DataFrame containing the loaded dataset with sentence1, sentence2, label columns

        Raises:
            ValueError: If dataset_name is not supported
            FileNotFoundError: If dataset cannot be loaded
        """
        # Get HuggingFace dataset name from metadata (handles alternate names)
        metadata = self.dataset_metadata[self.get_canonical_name(dataset_name)]
        hf_dataset_name = metadata.hf_dataset_name

        try:
            print(
                f"Loading dataset {hf_dataset_name} (split: {self.split}) from HuggingFace Hub..."
            )

            # Import here to avoid circular imports
            from datasets import load_dataset

            # Load dataset from HuggingFace
            # Try loading without config first
            try:
                hf_dataset = load_dataset(hf_dataset_name, split=self.split)
            except ValueError as e:
                # If it fails because config is missing, try with language configs
                if "Config name is missing" not in str(e):
                    raise

                # Get language configs from languages setting or metadata
                languages = self.config.get("languages", None)

                # Convert MTEB language codes to lowercase for dataset configs
                # E.g., "EN" -> "en", "DE" -> "de"
                if languages is None or languages == []:
                    # Try to get all languages from metadata
                    canonical_name = self.get_canonical_name(dataset_name)
                    if canonical_name in self.dataset_metadata:
                        metadata = self.dataset_metadata[canonical_name]
                        if metadata.is_non_en() and metadata.languages:
                            lang_configs = [lang.lower() for lang in metadata.languages]
                            logger.info(
                                f"Using all {len(lang_configs)} languages from metadata: {lang_configs}"
                            )
                        else:
                            # Monolingual dataset - default to "en"
                            lang_configs = ["en"]
                            logger.warning(
                                f"Dataset {dataset_name} requires a language config but languages is not set. "
                                f"Defaulting to 'en'"
                            )
                    else:
                        # No metadata - default to "en"
                        lang_configs = ["en"]
                        logger.warning(
                            f"Dataset {dataset_name} requires a language config but languages is not set. "
                            f"Defaulting to 'en'"
                        )
                else:
                    lang_configs = [lang.lower() for lang in languages]

                # Load dataset with language configs (multilingual datasets)
                logger.info(
                    f"Loading {dataset_name} with language configs: {lang_configs}"
                )
                datasets = []
                for config in lang_configs:
                    try:
                        ds = load_dataset(hf_dataset_name, config, split=self.split)
                        datasets.append(ds)
                        logger.info(f"  Loaded config '{config}' successfully")
                    except Exception as config_error:
                        logger.warning(
                            f"  Failed to load config '{config}': {config_error}"
                        )

                if not datasets:
                    raise ValueError(
                        f"Failed to load any language configs for {dataset_name}. "
                        f"Tried configs: {lang_configs}"
                    )

                # Merge datasets from different configs
                from datasets import concatenate_datasets

                hf_dataset = concatenate_datasets(datasets)
                logger.info(
                    f"Concatenated {len(datasets)} language configs for {dataset_name}"
                )

            # Convert to pandas DataFrame
            df = hf_dataset.to_pandas()

            df = self._normalize_column_names(df)

            # Check if dataset is in array format (1 row) or expanded format (multiple rows)
            if len(df) == 1:
                # Array format - extract arrays from single row
                sentences1_array = df.iloc[0]["sentence1"]
                sentences2_array = df.iloc[0]["sentence2"]
                labels_array = df.iloc[0]["label"]

                # Validate arrays have same length
                if not (
                    len(sentences1_array) == len(sentences2_array) == len(labels_array)
                ):
                    raise ValueError(
                        f"Array lengths don't match: sentence1={len(sentences1_array)}, "
                        f"sentence2={len(sentences2_array)}, labels={len(labels_array)}"
                    )

                # Create new DataFrame with expanded rows
                df = pd.DataFrame(
                    {
                        "sentence1": sentences1_array,
                        "sentence2": sentences2_array,
                        "label": labels_array,
                    }
                )

                print(f"Expanded {len(df)} sentence pairs from array structure")
            else:
                # Already in expanded format - just ensure column names are correct
                # Column might be 'labels' (plural) instead of 'label' (singular)
                if "labels" in df.columns and "label" not in df.columns:
                    df = df.rename(columns={"labels": "label"})
                print(f"Loaded {len(df)} sentence pairs in expanded format")

            # Convert labels to binary integers if they're not already
            if df["label"].dtype == "object":
                # Map string labels to binary integers
                unique_labels = df["label"].unique()
                if len(unique_labels) == 2:
                    label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
                    df["label"] = df["label"].map(label_mapping)
                    print(f"Mapped labels: {label_mapping}")
                else:
                    raise ValueError(
                        f"Expected binary labels, got {len(unique_labels)} unique values: {unique_labels}"
                    )

            # Validate that labels are 0 or 1
            unique_labels = set(df["label"].unique())
            if not unique_labels.issubset({0, 1}):
                raise ValueError(
                    f"Labels must be binary (0 or 1), got: {unique_labels}"
                )

            print(f"Loaded {len(df)} sentence pairs from {dataset_name}")
            print(f"Label distribution: {df['label'].value_counts().to_dict()}")

            # Apply downsampling if configured
            df = self.downsample_data(df)

            return df

        except Exception as e:
            print(f"Error loading dataset {hf_dataset_name}: {e}")
            raise

    def _get_metadata(
        self,
        emb_model_name: str,
        embedding_model: Any,
        gen_model_name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate common metadata dictionary for evaluation results.

        Args:
            emb_model_name: Name of the embedding model
            embedding_model: The embedding model object
            gen_model_name: Name of the generative model (can be None)
            **kwargs: Additional parameters

        Returns:
            Dictionary containing model metadata
        """
        return {
            "timestamp": self.timestamp,
            "run": kwargs.get("run", 0),
            "emb_model": emb_model_name,
            "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
            "model_max_seq_length": kwargs.get(
                "model_max_seq_length", getattr(embedding_model, "max_seq_length", None)
            ),
            "emb_batch_size": self.config.get("emb_batch_size", 32),
            "dtype": str(self.config.get("st_dtype", "torch.float32")),
            "encoding_chunk_size": self.config.get("encoding_chunk_size", None),
            "emb_model_api": self.config.get("emb_model_api", "sentence-transformers"),
            "gen_model": gen_model_name,
            "gen_api": "ollama" if gen_model_name else None,
            "paraphrase_prompt": self.pre_prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def _combine_results(
        self, original_result: Dict[str, Any], paraphrased_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine original and paraphrased evaluation results with prefixes.

        Args:
            original_result: Results from evaluating original sentences
            paraphrased_result: Results from evaluating paraphrased sentences

        Returns:
            Combined results dictionary with prefixed keys
        """
        result = {}

        # Add original results with prefix
        for key, value in original_result.items():
            result[f"original_{key}"] = value

        # Add paraphrased results with prefix
        for key, value in paraphrased_result.items():
            result[f"paraphrased_{key}"] = value

        return result

    def _extract_main_score(
        self, evaluation_result: Dict[str, Any], prefix: str
    ) -> float:
        """
        Extract main score (cosine AP) from evaluation result.

        Args:
            evaluation_result: Dictionary containing evaluation metrics
            prefix: Prefix for the result key (unused, kept for interface consistency)

        Returns:
            AP score value, or 0.0 if evaluation failed
        """
        if "error" in evaluation_result:
            return 0.0
        return evaluation_result["cosine_ap"]

    def evaluate(
        self,
        dataset_name: str,
        embedding_model: Any,
        emb_model_name: str,
        data: pd.DataFrame | None = None,
        gen_model_name: str | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate the embedding model on pair classification data with paraphrases.

        Args:
            dataset_name: Name of the dataset being evaluated
            embedding_model: The embedding model to evaluate
            emb_model_name: Name of the embedding model
            data: DataFrame with the dataset (if None, will load from dataset_name)
            gen_model_name: Name of generative model for paraphrases
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation results
        """
        if self.gen_model:
            return self._evaluate_with_online_paraphrases(
                dataset_name,
                embedding_model,
                emb_model_name,
                data,
                gen_model_name,
                **kwargs,
            )
        else:
            return self._evaluate_original_only(
                dataset_name, embedding_model, emb_model_name, data, **kwargs
            )

    def _evaluate_with_online_paraphrases(
        self,
        dataset_name: str,
        embedding_model: Any,
        emb_model_name: str,
        data: pd.DataFrame,
        gen_model_name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate with paraphrase generation."""
        print(f"Generating paraphrases on-demand for {gen_model_name}...")

        # Load data if not provided
        if data is None:
            data = self.load_hf_data(dataset_name)

        # Extract sentences and labels
        sentences1 = data["sentence1"].tolist()
        sentences2 = data["sentence2"].tolist()
        labels = data["label"].tolist()

        # Get seed from kwargs for Type 1 architecture
        seed = kwargs.get("seed", self.seed_paraphrasing)
        paraphrases1, paraphrases2 = self.paraphrase_data(
            sentences1=sentences1,
            sentences2=sentences2,
            gen_model_name=gen_model_name,
            gen_api_name="ollama",
            dataset_name=dataset_name,
            seed=seed,
        )

        # Save paraphrases if requested
        if self.save_paraphrases:
            seeds_used = [seed] * len(sentences1)
            self._save_paraphrases(
                original_sentences1=sentences1,
                paraphrased_sentences1=paraphrases1,
                original_sentences2=sentences2,
                paraphrased_sentences2=paraphrases2,
                labels=labels,
                gen_model_name=gen_model_name,
                gen_api_name="ollama",
                dataset_name=dataset_name,
                seeds_used=seeds_used,
            )

        # Evaluate original and paraphrased sentences
        print(f"=== Evaluating Original Sentences ({len(sentences1)} pairs) ===")
        original_result = self.evaluator(
            embedding_model,
            sentences1=sentences1,
            sentences2=sentences2,
            labels=labels,
            phase="Original",
        )

        print(f"=== Evaluating Paraphrased Sentences ({len(paraphrases1)} pairs) ===")
        paraphrased_result = self.evaluator(
            embedding_model,
            sentences1=paraphrases1,
            sentences2=paraphrases2,
            labels=labels,
            phase="Paraphrased",
        )

        # Combine results
        result = self._combine_results(original_result, paraphrased_result)

        # Add main scores
        original_main = self._extract_main_score(original_result, "original")
        if original_main is not None:
            result["original_main_score"] = original_main

        paraphrased_main = self._extract_main_score(paraphrased_result, "paraphrased")
        if paraphrased_main is not None:
            result["paraphrased_main_score"] = paraphrased_main

        # Add metadata
        metadata = self._get_metadata(
            emb_model_name, embedding_model, gen_model_name, **kwargs
        )
        result.update(metadata)

        return result

    def _evaluate_original_only(
        self,
        dataset_name: str,
        embedding_model: Any,
        emb_model_name: str,
        data: pd.DataFrame,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate only original sentences (no generative models)."""
        # Load data if not provided
        if data is None:
            data = self.load_hf_data(dataset_name)

        # Extract sentences and labels
        sentences1 = data["sentence1"].tolist()
        sentences2 = data["sentence2"].tolist()
        labels = data["label"].tolist()

        print(
            f"No generative models configured. Evaluating {len(sentences1)} original sentence pairs..."
        )
        result = self.evaluator(
            embedding_model,
            sentences1=sentences1,
            sentences2=sentences2,
            labels=labels,
        )

        # Add metadata (no generative model info)
        metadata = self._get_metadata(emb_model_name, embedding_model, None, **kwargs)
        result.update(metadata)

        return result

    def _get_evaluation_metadata(
        self,
        dataset_name: str,
        n_samples: int,
        paraphrase_times: list[float] | None = None,
        gen_model: str | None = None,
        run_seed: int | None = None,
    ) -> Dict[str, Any]:
        """
        Get evaluation metadata for results.

        Args:
            dataset_name: Name of the dataset
            n_samples: Number of samples
            paraphrase_times: List of paraphrase generation times (unused)
            gen_model: Generative model name (unused)
            run_seed: Run-specific seed (if None, uses self.seed_paraphrasing)

        Returns:
            Dictionary containing evaluation metadata
        """
        # Use run-specific seed if provided, otherwise use base seed
        seed_for_paraphrasing = (
            run_seed if run_seed is not None else self.seed_paraphrasing
        )

        metadata = {
            "dataset": self.get_canonical_name(dataset_name),
            "n_samples": n_samples,
            "timestamp": self.timestamp,
            "task": self.task_name,
            "runtime": 0.0,  # Will be updated by caller if needed
            "seed_general": self.seed_general,
            "seed_paraphrasing": seed_for_paraphrasing,
        }
        return metadata

    def run_full_evaluation(
        self, embedding_models: List[Any], model_names: List[str], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run complete evaluation across all datasets, embedding models, and generative models.

        Args:
            embedding_models: List of embedding models to evaluate
            model_names: List of model names corresponding to embedding_models
            **kwargs: Additional evaluation parameters

        Returns:
            List of all evaluation results
        """
        all_results = []

        for dataset_name in self.datasets:
            print(f"\n{'=' * 50}")
            print(f"Evaluating dataset: {dataset_name}")
            print(f"{'=' * 50}")

            # Load data
            data = self.load_hf_data(dataset_name)

            # Paraphrases will be generated per run for dataset
            if self.gen_model:
                print(
                    f"\nParaphrases will be generated per run for dataset: {dataset_name}"
                )

            # Evaluate with each embedding model
            for model_idx, emb_model in enumerate(embedding_models):
                emb_model_name = model_names[model_idx]
                print(f"\nEvaluating with embedding model: {emb_model_name}")

                if self.gen_model:
                    # Evaluate with generative model using Type 1 architecture (run loop)
                    # Add run loop for Type 1 architecture
                    for run in range(self.n_runs):
                        run_seed = self.get_seed_for_run(run)
                        print(
                            f"  Using generative model: {self.gen_model} (ollama) - Run {run + 1}/{self.n_runs} (seed: {run_seed})"
                        )

                        result = self.evaluate(
                            dataset_name=dataset_name,
                            embedding_model=emb_model,
                            emb_model_name=emb_model_name,
                            data=data,
                            gen_model_name=self.gen_model,
                            seed=run_seed,
                            run=run,
                            **kwargs,
                        )

                        # Handle case where evaluate returns a list
                        if isinstance(result, list):
                            # Add run information to each result
                            for r in result:
                                r["run"] = run + 1
                            all_results.extend(result)
                        else:
                            metadata = self._get_evaluation_metadata(
                                dataset_name, len(data), run_seed=run_seed
                            )
                            metadata["run"] = run + 1
                            result.update(metadata)
                            all_results.append(result)
                else:
                    # Evaluate without generative models
                    result = self.evaluate(
                        dataset_name=dataset_name,
                        embedding_model=emb_model,
                        emb_model_name=emb_model_name,
                        data=data,
                        **kwargs,
                    )
                    metadata = self._get_evaluation_metadata(dataset_name, len(data))
                    result.update(metadata)
                    all_results.append(result)

                # Clean up embedding model after evaluation to free GPU memory
                cleanup_model(emb_model, emb_model_name)

        return all_results

    def get_summary_table(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a summary table from evaluation results.

        Args:
            results: List of evaluation result dictionaries

        Returns:
            DataFrame with key metrics for display
        """
        if not results:
            return pd.DataFrame()

        # Select key columns for summary
        summary_cols = [
            "dataset",
            "emb_model",
            "gen_model",
            "original_cosine_accuracy",
            "paraphrased_cosine_accuracy",
            "original_cosine_f1",
            "paraphrased_cosine_f1",
            "original_cosine_ap",
            "paraphrased_cosine_ap",
            "original_main_score",
            "paraphrased_main_score",
            "n_samples",
            "runtime",
        ]

        # Filter available columns
        available_cols = [col for col in summary_cols if col in results[0]]

        df = pd.DataFrame(results)

        return df[available_cols] if available_cols else df

    def paraphrase_data(
        self,
        sentences1: list[str] | None = None,
        sentences2: list[str] | None = None,
        gen_model_name: str | None = None,
        gen_api_name: str | None = None,
        dataset_name: str | None = None,
        seed: int | None = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Override to use pair classification-specific paraphrase generation with saving.

        This ensures that paraphrases are saved when save_paraphrases=true.
        """
        # Use single-run approach
        actual_seed = seed if seed is not None else self.seed_paraphrasing
        paraphrases1 = self._generate_paraphrases(
            texts=sentences1,
            gen_model_name=gen_model_name,
            gen_api_name="ollama",
            dataset_name=dataset_name,
            seed=actual_seed,
        )
        paraphrases2 = self._generate_paraphrases(
            texts=sentences2,
            gen_model_name=gen_model_name,
            gen_api_name="ollama",
            dataset_name=dataset_name,
            seed=actual_seed,
        )
        return paraphrases1, paraphrases2

    def _save_paraphrases(
        self,
        original_sentences1: List[str],
        paraphrased_sentences1: List[str],
        original_sentences2: List[str],
        paraphrased_sentences2: List[str],
        labels: List[Union[int, float, None]],
        gen_model_name: str,
        gen_api_name: str,
        dataset_name: str,
        seeds_used: Optional[List[Optional[int]]] = None,
        split_override: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ) -> None:
        """
        Save generated paraphrases to disk with pair classification specific format.

        Args:
            original_sentences1: Original first sentences
            paraphrased_sentences1: Paraphrased first sentences
            original_sentences2: Original second sentences
            paraphrased_sentences2: Paraphrased second sentences
            labels: Classification labels (0/1) or None
            gen_model_name: Name of the generative model
            gen_api_name: API name for the generative model (hardcoded to "ollama")
            dataset_name: Name of the dataset being processed
            seeds_used: Optional list of seeds used for each paraphrase
            split_override: Override the split name used in the file path
            languages: Optional list of language codes for each sentence pair
        """
        import os
        import re

        from ..utils import save_df

        # Create DataFrame with pair classification specific column names
        paraphrase_df = pd.DataFrame(
            {
                "original_sentence1": original_sentences1,
                "paraphrase_sentence1": paraphrased_sentences1,
                "original_sentence2": original_sentences2,
                "paraphrase_sentence2": paraphrased_sentences2,
                "label": labels,
            }
        )

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
            "pair_classification",
            dataset_name_for_path,
            effective_split,
            re.sub(r":", "-", gen_model_name),
        )

        os.makedirs(para_output_dir, exist_ok=True)

        # Create filename with timestamp
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

        # Use JSONL format for consistency
        output_file = os.path.join(para_output_dir, para_filename)

        # Save using the utility function
        save_df(paraphrase_df, output_file, file_format="jsonl")

        print(f"Saved pair classification paraphrases to {output_file}")

    def _average_results(self, results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Average results across multiple runs.

        Args:
            results_list: List of result dictionaries from multiple runs

        Returns:
            Dictionary with averaged metrics
        """
        if not results_list:
            raise ValueError("Cannot average empty results list")

        if len(results_list) == 1:
            return results_list[0]

        averaged_result = {}

        # Get all numeric keys from the first result
        numeric_keys = []
        for key, value in results_list[0].items():
            if isinstance(value, (int, float)):
                numeric_keys.append(key)

        # Average each numeric metric
        for key in numeric_keys:
            values = [result[key] for result in results_list if key in result]
            if values:
                averaged_result[key] = sum(values) / len(values)

        # Keep non-numeric values from the first result
        for key, value in results_list[0].items():
            if key not in numeric_keys:
                averaged_result[key] = value

        return averaged_result
