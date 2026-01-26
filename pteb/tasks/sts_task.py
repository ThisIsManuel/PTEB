"""
STS (Semantic Textual Similarity) task implementation for PTEB.

This module implements the STSTask class that handles STS dataset loading,
paraphrase generation/loading, and evaluation with correlation metrics.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from datasets import load_dataset

from ..evaluators.sts_evaluator import STSEvaluator
from ..utils import (
    apply_language_filter,
    cleanup_model,
    save_df,
    should_use_multilingual_suffix,
)
from .paraphrase_task import ParaphraseTask


class STSTask(ParaphraseTask):
    """
    Semantic Textual Similarity task with paraphrase-enhanced evaluation.

    This class handles:
    - Loading STS datasets from HuggingFace Hub
    - Generating or loading paraphrases
    - Evaluating embedding models on original and paraphrased data
    - Computing Spearman and Pearson correlations
    """

    def downsample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downsample sentence pairs DataFrame."""
        downsample = self.config.get("downsample")
        if downsample is None or downsample >= 1.0:
            return df

        n_sample = max(1, int(len(df) * downsample))
        df_sampled = df.sample(n=n_sample, random_state=self.seed_general)

        print(f"Downsampled to {n_sample} pairs ({downsample*100:.1f}%)")
        return df_sampled

    def _handle_oom_error(
        self,
        dataset_name: str,
        emb_model_name: str,
        gen_model_name: str,
        n_samples: int,
        error_context: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Handle CUDA OOM errors with standardized error response.

        Args:
            dataset_name: Name of the dataset
            emb_model_name: Name of the embedding model
            gen_model_name: Name of the generative model (can be None)
            n_samples: Number of samples being processed
            error_context: Description of when the error occurred
            **kwargs: Additional parameters (e.g., model_vram_gb)

        Returns:
            Standardized error response dictionary
        """
        print(
            f"ERROR: {error_context} failed due to CUDA OOM for dataset {dataset_name}"
        )
        return {
            "dataset": self.get_canonical_name(dataset_name),
            "task": "STS",
            "emb_model": emb_model_name,
            "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
            "gen_model": gen_model_name,
            "error": "CUDA_OOM",
            "error_message": f"CUDA out of memory during {error_context.lower()}",
            "n_samples": n_samples,
            "spearman": 0.0,
            "pearson": 0.0,
            "main_score": 0.0,
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

        # Add main scores for convenience
        # The evaluator returns metric_spearman (e.g., cosine_spearman)
        # Extract the spearman value regardless of the metric prefix
        spearman_keys = [k for k in original_result.keys() if k.endswith("_spearman")]
        if spearman_keys:
            result["original_main_score"] = original_result[spearman_keys[0]]

        spearman_keys = [
            k for k in paraphrased_result.keys() if k.endswith("_spearman")
        ]
        if spearman_keys:
            result["paraphrased_main_score"] = paraphrased_result[spearman_keys[0]]

        return result

    def _get_metadata(
        self,
        emb_model_name: str,
        embedding_model: Any,
        gen_model_name: str,
        gen_api_name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate common metadata dictionary for evaluation results.

        Args:
            emb_model_name: Name of the embedding model
            embedding_model: The embedding model object
            gen_model_name: Name of the generative model (can be None)
            gen_api_name: API name for generative model (can be None)
            **kwargs: Additional parameters

        Returns:
            Dictionary containing model metadata
        """
        return {
            "emb_model": emb_model_name,
            "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
            "model_max_seq_length": kwargs.get(
                "model_max_seq_length", getattr(embedding_model, "max_seq_length", None)
            ),
            "emb_batch_size": self.config.get("emb_batch_size", 32),
            "dtype": str(self.config.get("st_dtype", "torch.float32")),
            "emb_model_api": self.config.get("emb_model_api", "sentence-transformers"),
            "encoding_chunk_size": self.config.get("encoding_chunk_size", None),
            "gen_model": gen_model_name,
            "gen_api": gen_api_name,
            "paraphrase_prompt": self.pre_prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

    def _extract_and_normalize_scores(
        self, data: pd.DataFrame, dataset_name: str
    ) -> Optional[List[float]]:
        """
        Extract scores from data and normalize them.

        Args:
            data: DataFrame containing the scores
            dataset_name: Name of the dataset for normalization

        Returns:
            Normalized scores or None if no valid scores available
        """
        if data is None or "score" not in data.columns:
            return None

        raw_scores = data["score"].tolist()
        if any(score is not None for score in raw_scores):
            return self.normalize_scores(raw_scores, dataset_name)
        return None

    def normalize_scores(
        self, scores: List[Union[float, None]], dataset_name: str
    ) -> List[float]:
        """
        Normalize STS scores to 0-1 range using min-max scaling.
        Filters out None values before returning.

        Args:
            scores: List of raw STS scores from the dataset
            dataset_name: Name of the dataset (normalized) to get score range

        Returns:
            List of normalized scores in [0, 1] range (None values filtered out)
        """
        # Get score range from dataset metadata
        canonical_name = self.get_canonical_name(dataset_name)
        if canonical_name not in self.dataset_metadata:
            print(
                f"Warning: No metadata found for dataset '{dataset_name}', using scores as-is"
            )
            # Filter out None values even when not normalizing
            return [
                float(score)
                for score in scores
                if score is not None and not pd.isna(score)
            ]

        metadata = self.dataset_metadata[canonical_name]
        if metadata.score_range is None:
            print(
                f"Warning: No score range defined for dataset '{dataset_name}', using scores as-is"
            )
            # Filter out None values even when not normalizing
            return [
                float(score)
                for score in scores
                if score is not None and not pd.isna(score)
            ]

        min_score, max_score = metadata.score_range

        # Apply min-max normalization: (x - min) / (max - min)
        normalized_scores = []
        for score in scores:
            if score is not None and not pd.isna(score):
                # Clamp score to valid range first
                clamped_score = max(min_score, min(max_score, float(score)))
                normalized = (clamped_score - min_score) / (max_score - min_score)
                normalized_scores.append(normalized)

        print(
            f"Normalized {len(normalized_scores)} scores for {dataset_name} from range [{min_score}, {max_score}] to [0, 1]"
        )
        return normalized_scores

    def __init__(self, datasets: List[str], config: Dict[str, Any], **kwargs):
        """
        Initialize the STS task.

        Args:
            datasets: List of STS dataset names to evaluate on
            config: Configuration dictionary from CLI arguments
            **kwargs: Additional parameters
        """
        super().__init__(
            task_name="STS",
            task_description="Semantic Textual Similarity evaluation with paraphrase augmentation",
            datasets=datasets,
            config=config,
            **kwargs,
        )

        # Initialize STS evaluator with score normalization support
        # Note: Score range will be set per dataset during evaluation
        # Pass seed_general for ML/evaluation operations (not paraphrasing)
        config_with_seed = {**config, "seed": self.seed_general}
        self.evaluator = STSEvaluator(**config_with_seed)

    def load_hf_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load and return the specified STS dataset from HuggingFace Hub.

        Args:
            dataset_name: Name of the STS dataset to load

        Returns:
            DataFrame containing the loaded dataset with sentence1, sentence2, score columns

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

            # Load dataset from HuggingFace
            hf_dataset = load_dataset(hf_dataset_name, split=self.split)

            # Convert to pandas DataFrame
            df = hf_dataset.to_pandas()

            # Ensure required columns exist
            required_cols = ["sentence1", "sentence2"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns: {missing_cols}. Available: {list(df.columns)}"
                )

            # Apply centralized language filtering
            df = apply_language_filter(df, self.config)

            # Apply downsampling if configured
            df = self.downsample_data(df)

            print(f"Loaded {len(df)} sentence pairs from {dataset_name}")

            return df

        except Exception as e:
            print(f"Error loading dataset {hf_dataset_name}: {e}")
            raise

    def paraphrase_data(
        self,
        sentences1: list[str] | None = None,
        sentences2: list[str] | None = None,
        gen_model_name: str | None = None,
        gen_api_name: str | None = None,
        dataset_name: str | None = None,
        seed: int | None = None,
        languages: list[str] | None = None,
        scores: Optional[List[float]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Override to use STS-specific paraphrase generation with saving.

        This ensures that paraphrases are saved when save_paraphrases=true.
        """
        # Use STS-specific method that includes saving logic for single run
        actual_seed = seed if seed is not None else self.seed_paraphrasing
        return self.generate_paraphrases(
            sentences1,
            sentences2,
            gen_model_name,
            gen_api_name,
            scores,
            actual_seed,
            dataset_name,
            languages,
        )

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
        Evaluate the embedding model on STS data with paraphrases.

        Args:
            dataset_name: Name of the dataset being evaluated
            embedding_model: The embedding model to evaluate
            data: DataFrame with the dataset (if None, will load from dataset_name)
            gen_model_name: Name of generative model for paraphrases
            gen_api_name: API name for generative model
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
                gen_api_name,
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
        gen_api_name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate with paraphrase generation."""
        print(f"Generating paraphrases on-demand for {gen_model_name}...")

        if data is None:
            data = self.load_hf_data(dataset_name)

        sentences1 = data["sentence1"].tolist()
        sentences2 = data["sentence2"].tolist()

        # Extract raw scores for saving (will be normalized later for evaluation)
        raw_scores = data["score"].tolist() if "score" in data.columns else None

        # For evaluation, we need normalized scores
        scores = self._extract_and_normalize_scores(data, dataset_name)

        # Extract language information if available
        languages = None
        if "lang" in data.columns:
            languages = data["lang"].tolist()

        # Pass seed from kwargs for Type 1 architecture
        seed = kwargs.get("seed_paraphrasing", self.seed_paraphrasing)
        paraphrases1, paraphrases2 = self.paraphrase_data(
            sentences1=sentences1,
            sentences2=sentences2,
            gen_model_name=gen_model_name,
            gen_api_name=gen_api_name,
            dataset_name=dataset_name,
            seed=seed,
            languages=languages,
            scores=raw_scores,  # Pass raw scores for saving, not normalized
        )

        return self._evaluate_seeds(
            embedding_model,
            sentences1,
            sentences2,
            paraphrases1,
            paraphrases2,
            scores,
            None,  # No seed_ids for online mode
            dataset_name,
            emb_model_name,
            gen_model_name,
            gen_api_name,
            **kwargs,
        )

    def _evaluate_original_only(
        self,
        dataset_name: str,
        embedding_model: Any,
        emb_model_name: str,
        data: pd.DataFrame,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate only original sentences (no generative models)."""
        if data is None:
            data = self.load_hf_data(dataset_name)

        sentences1 = data["sentence1"].tolist()
        sentences2 = data["sentence2"].tolist()
        scores = self._extract_and_normalize_scores(data, dataset_name)

        individual_eval_start = time.time()
        result = self.evaluator(
            embedding_model, sentences1=sentences1, sentences2=sentences2, scores=scores
        )
        individual_eval_time = time.time() - individual_eval_start

        if result.get("error") == "CUDA_OOM":
            return self._handle_oom_error(
                dataset_name,
                emb_model_name,
                None,
                len(sentences1),
                "Evaluation",
                **kwargs,
            )

        metadata = self._get_metadata(
            emb_model_name, embedding_model, None, None, **kwargs
        )
        metadata["individual_eval_time"] = individual_eval_time
        result.update(metadata)
        return result

    def _evaluate_seeds(
        self,
        embedding_model: Any,
        sentences1: List[str],
        sentences2: List[str],
        paraphrases1: List[str],
        paraphrases2: List[str],
        scores: Optional[List[float]],
        seed_ids: Optional[List[int]],
        dataset_name: str,
        emb_model_name: str,
        gen_model_name: str,
        gen_api_name: str,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Evaluate original and paraphrased sentences, handling both single and multiple seeds."""
        print("=== Evaluating Original Sentences ===")
        original_result = self.evaluator(
            embedding_model,
            sentences1=sentences1,
            sentences2=sentences2,
            scores=scores,
            phase="Original",
        )

        if original_result.get("error") == "CUDA_OOM":
            error_result = self._handle_oom_error(
                dataset_name,
                emb_model_name,
                gen_model_name,
                len(sentences1),
                "Original evaluation",
                **kwargs,
            )
            return (
                [error_result] if seed_ids and len(set(seed_ids)) > 1 else error_result
            )

        # Handle seed grouping - create artificial seed_ids if none provided
        if seed_ids is None:
            seed_ids = [1] * len(paraphrases1)  # Single seed for all

        unique_seeds = sorted(set(seed_ids))

        # Group paraphrases by seed
        df = pd.DataFrame(
            {
                "seed": seed_ids,
                "para_s1": paraphrases1,
                "para_s2": paraphrases2,
                "scores": scores,
            }
        )
        per_seed_results = []

        if len(unique_seeds) > 1:
            print(f"=== Evaluating {len(unique_seeds)} seeds separately ===")
        else:
            print("=== Evaluating Paraphrased Sentences ===")

        for seed_id in unique_seeds:
            seed_data = df[df["seed"] == seed_id]
            seed_para_s1 = seed_data["para_s1"].tolist()
            seed_para_s2 = seed_data["para_s2"].tolist()
            seed_scores = seed_data["scores"].tolist()

            if len(unique_seeds) > 1:
                print(f"Evaluating seed {seed_id} ({len(seed_para_s1)} samples)")
                phase = f"Paraphrased (seed {seed_id})"
            else:
                phase = "Paraphrased"

            paraphrased_result = self.evaluator(
                embedding_model,
                sentences1=seed_para_s1,
                sentences2=seed_para_s2,
                scores=seed_scores,
                phase=phase,
            )

            if paraphrased_result.get("error") == "CUDA_OOM":
                error_context = (
                    f"Paraphrased evaluation (seed {seed_id})"
                    if len(unique_seeds) > 1
                    else "Paraphrased evaluation"
                )
                error_result = self._handle_oom_error(
                    dataset_name,
                    emb_model_name,
                    gen_model_name,
                    len(sentences1),
                    error_context,
                    **kwargs,
                )
                return [error_result] if len(unique_seeds) > 1 else error_result

            seed_result = self._combine_results(original_result, paraphrased_result)

            # Add actual sample count used for this seed
            seed_result["n_samples"] = len(seed_para_s1)

            if len(unique_seeds) > 1:
                seed_result["run"] = seed_id
            per_seed_results.append(seed_result)

        # Add metadata to all results
        metadata = self._get_metadata(
            emb_model_name, embedding_model, gen_model_name, gen_api_name, **kwargs
        )
        for r in per_seed_results:
            r.update(metadata)

        # Return list for multiple seeds, single dict for single seed
        return per_seed_results if len(unique_seeds) > 1 else per_seed_results[0]

    def _get_evaluation_metadata(self, dataset_name, n_samples, gen_model=None):
        """
        Create evaluation metadata dictionary.

        Note: n_samples is NOT included in the metadata as it's added directly in the
        evaluation method based on actual filtered data.

        Args:
            dataset_name: Name of dataset
            n_samples: Number of samples (not used, kept for compatibility)
            gen_model: Generative model name (optional)

        Returns:
            Metadata dictionary
        """
        return {
            "dataset": self.get_canonical_name(dataset_name),
            # n_samples excluded - added directly in _evaluate_seeds with actual count
            "timestamp": self.timestamp,
            "task": self.task_name,
            "main_metric": self.evaluator.get_main_score_name(),
            "runtime": 0.0,  # Will be updated by caller if needed
            "seed_general": self.seed_general,
            "seed_paraphrasing": self.seed_paraphrasing,
        }

    def _display_main_scores(self, all_results):
        """
        Display main scores for the current evaluation results.

        Args:
            all_results: List of evaluation results
        """
        if len(all_results) > 0 and isinstance(all_results[-1], list):
            # Multiple results per seed (last result is a list)
            for i, result in enumerate(all_results[-1]):
                if "main_score" in result:
                    print(f"    Seed {i + 1} Main Score: {result['main_score']:.4f}")
        elif len(all_results) > 0:
            # Single result (check the last result)
            result = all_results[-1]
            if "main_score" in result:
                print(f"    Main Score: {result['main_score']:.4f}")

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

            # Load and prepare data
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
                            gen_api_name="ollama",
                            seed_paraphrasing=run_seed,
                            run=run,
                            **kwargs,
                        )

                        n_samples = len(data)
                        metadata = self._get_evaluation_metadata(
                            dataset_name, n_samples, self.gen_model
                        )
                        # Add run information to metadata
                        metadata["run"] = run + 1
                        metadata["seed_paraphrasing"] = run_seed

                        if isinstance(result, list):
                            for r in result:
                                r.update(metadata)
                            all_results.extend(result)
                        else:
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

                    n_samples = len(data) if data is not None else 0
                    metadata = self._get_evaluation_metadata(dataset_name, n_samples)
                    result.update(metadata)
                    all_results.append(result)

                self._display_main_scores(all_results)

        # Clean up embedding models
        for model_idx, emb_model in enumerate(embedding_models):
            emb_model_name = model_names[model_idx]
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
            "original_spearman",
            "paraphrased_spearman",
            "n_samples",
            "runtime",
        ]

        # Filter available columns
        available_cols = [col for col in summary_cols if col in results[0]]

        df = pd.DataFrame(results)

        # Replace long dataset names for cleaner display
        if "dataset" in df.columns:
            df["dataset"] = df["dataset"].replace("stsbenchmark", "STSB")

        return df[available_cols] if available_cols else df

    def _save_paraphrases(
        self,
        sentences1: List[str],
        paraphrases1: List[str],
        sentences2: List[str],
        paraphrases2: List[str],
        scores: Optional[List[float]],
        gen_model_name: str,
        gen_api_name: str,
        dataset_name: str,
        seeds_used: Optional[List[Optional[int]]] = None,
        split_override: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ) -> None:
        """
        Save generated paraphrases to disk with STS-specific format.

        Args:
            sentences1: Original first sentences
            paraphrases1: Paraphrased first sentences
            sentences2: Original second sentences
            paraphrases2: Paraphrased second sentences
            scores: Optional similarity scores
            gen_model_name: Name of the generative model
            gen_api_name: API name for the generative model
            dataset_name: Name of the dataset being processed
            seeds_used: Optional list of seeds used for each paraphrase
            split_override: Override the split name used in the file path
        """

        # Create DataFrame with STS-specific column names
        paraphrase_df = pd.DataFrame(
            {
                "original_sentence1": sentences1,
                "paraphrase_sentence1": paraphrases1,
                "original_sentence2": sentences2,
                "paraphrase_sentence2": paraphrases2,
            }
        )

        if scores:
            paraphrase_df["score"] = scores

        # Add language information if available
        if languages:
            paraphrase_df["language"] = languages
        else:
            # Default to English if no language info available
            paraphrase_df["language"] = ["en"] * len(sentences1)

        # Determine output directory
        effective_split = split_override if split_override is not None else self.split

        # Use canonical name for path, add multilingual suffix if needed (only for STS17/STS22)
        dataset_name_for_path = self.get_canonical_name(dataset_name)
        if should_use_multilingual_suffix(dataset_name_for_path, self.config):
            dataset_name_for_path = f"{dataset_name_for_path}_multilingual"

        para_output_dir = os.path.join(
            "data",
            "paraphrases",
            "sts",
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
        para_filename = re.sub(r":", "-", para_filename)

        para_output_path = os.path.join(para_output_dir, para_filename)

        # Save paraphrases in JSONL format for consistency
        output_file = para_output_path
        save_df(paraphrase_df, output_file, file_format="jsonl")

        print(f"STS paraphrases saved to {output_file}")

    def generate_paraphrases(
        self,
        sentences1: List[str],
        sentences2: List[str],
        gen_model: str,
        gen_api: str,
        scores: Optional[List[float]] = None,
        seed: int = 1337,
        dataset_name: str | None = None,
        languages: list[str] | None = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate paraphrases for STS sentence pairs using single-run approach.

        Args:
            sentences1: List of first sentences
            sentences2: List of second sentences
            gen_model: Name of generative model
            gen_api: API name for generative model
            scores: Optional similarity scores
            seed: Random seed for reproducibility
            dataset_name: Name of dataset (for saving)

        Returns:
            Tuple of (paraphrased_sentences1, paraphrased_sentences2)
        """
        # Generate paraphrases using single-run approach (Type 1 architecture)
        paraphrases1 = self._generate_paraphrases(
            texts=sentences1,
            gen_model_name=gen_model,
            gen_api_name=gen_api,
            dataset_name=dataset_name,
            seed=seed,
        )

        paraphrases2 = self._generate_paraphrases(
            texts=sentences2,
            gen_model_name=gen_model,
            gen_api_name=gen_api,
            dataset_name=dataset_name,
            seed=seed,
        )

        # Save paraphrases with STS format (only if saving is enabled)
        if self.save_paraphrases:
            seeds_used = [seed] * len(sentences1) if seed is not None else None
            self._save_paraphrases(
                sentences1=sentences1,
                paraphrases1=paraphrases1,
                sentences2=sentences2,
                paraphrases2=paraphrases2,
                scores=scores,
                gen_model_name=gen_model,
                gen_api_name=gen_api,
                dataset_name=dataset_name,
                seeds_used=seeds_used,
                languages=languages,
            )

        return paraphrases1, paraphrases2
