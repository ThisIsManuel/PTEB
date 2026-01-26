"""
Classification task implementation for PTEB.
"""

import logging
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import load_dataset

from ..evaluators.classification_evaluator import ClassificationEvaluator
from ..utils import should_use_multilingual_suffix
from .paraphrase_task import ParaphraseTask

logger = logging.getLogger(__name__)


class ClassificationTask(ParaphraseTask):
    """
    Task for evaluating classification performance using MTEB datasets.

    This task loads classification datasets, optionally generates paraphrases,
    and evaluates using LogisticRegression with accuracy as the main metric.
    """

    def __init__(self, datasets: List[str], config: Dict[str, Any]):
        """
        Initialize the classification task.

        Args:
            datasets: List of classification dataset names
            config: Configuration dictionary
        """
        # Initialize parent class with task metadata
        super().__init__(
            task_name="Classification",
            task_description="Multi-class text classification using LogisticRegression",
            datasets=datasets,
            config=config,
        )

        # Remove conflicting keys from config copy to avoid duplicate keyword arguments
        evaluator_config = dict(config)
        evaluator_config.pop("max_iter", None)
        evaluator_config.pop("seed", None)

        self.evaluator = ClassificationEvaluator(
            max_iter=config.get("max_iter", 100),
            seed=self.seed_general,
            **evaluator_config,
        )

        logger.info(f"ClassificationTask initialized with datasets: {self.datasets}")

    def downsample_data(
        self,
        data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Downsample train and test splits with stratified sampling.

        Samples proportionally from each label class to preserve label distribution.
        Guarantees at least 1 sample per label.
        """
        downsample = self.config.get("downsample")
        if downsample is None or downsample >= 1.0:
            return data

        from collections import defaultdict

        rng = random.Random(self.seed_general)
        result = {}

        for split_name, split_data in data.items():
            texts = split_data["texts"]
            labels = split_data["labels"]

            # Group indices by label for stratified sampling
            label_to_indices: Dict[Any, List[int]] = defaultdict(list)
            for i, label in enumerate(labels):
                label_to_indices[label].append(i)

            # Sample from each label class proportionally
            sampled_indices = []
            for label, indices in label_to_indices.items():
                n_sample_label = max(1, int(len(indices) * downsample))
                sampled = rng.sample(indices, min(n_sample_label, len(indices)))
                sampled_indices.extend(sampled)

            # Shuffle to avoid label ordering
            rng.shuffle(sampled_indices)

            result[split_name] = {
                "texts": [texts[i] for i in sampled_indices],
                "labels": labels[sampled_indices],
            }
            n_labels = len(label_to_indices)
            print(
                f"Downsampled {split_name} to {len(sampled_indices)} samples "
                f"({downsample*100:.1f}%), {n_labels} labels preserved"
            )

        return result

    def evaluate_dataset(
        self,
        dataset_name: str,
        embedding_model: Any,
        model_name: str,
        seed: int,
        gen_model: Optional[str] = None,
        run: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate a single classification dataset.

        Args:
            dataset_name: Name of the dataset to evaluate
            embedding_model: Embedding model to use
            model_name: Name of the embedding model
            gen_model: Optional generative model for paraphrases
            run: Run number for multiple evaluations
            seed: Random seed

        Returns:
            Dictionary with evaluation results
        """
        split = self.config.get("split", "test")

        # Load dataset
        data = self.load_hf_data(dataset_name, split)
        train_texts = data["train"]["texts"]
        train_labels = data["train"]["labels"]
        test_texts = data[split]["texts"]
        test_labels = data[split]["labels"]

        result = {
            "timestamp": self.timestamp,
            "dataset": self.get_canonical_name(dataset_name),
            "split": split,
            "emb_model": model_name,
            "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
            "model_max_seq_length": kwargs.get(
                "model_max_seq_length", getattr(embedding_model, "max_seq_length", None)
            ),
            "emb_model_api": self.config.get("emb_model_api", "sentence-transformers"),
            "emb_batch_size": self.config.get("emb_batch_size"),
            "dtype": str(self.config.get("st_dtype")),
            "encoding_chunk_size": self.config.get("encoding_chunk_size", None),
            "n_train_samples": len(train_texts),
            "n_test_samples": len(test_texts),
            "n_samples": len(test_texts),  # For consistency with other tasks
            "task": "Classification",
            "run": run,
            "seed_general": self.seed_general,
            "seed_paraphrasing": seed,
        }

        if gen_model is None:
            # Original evaluation only
            logger.info(f"Evaluating {dataset_name} with original data only")

            # Evaluate using the evaluator's __call__ method
            start_time = time.time()
            scores = self.evaluator(
                embedding_model,
                train_sentences=train_texts,
                train_labels=train_labels,
                test_sentences=test_texts,
                test_labels=test_labels,
            )
            embedding_time = time.time() - start_time

            result.update(scores)
            result["embedding_generation_time"] = embedding_time
            result["gen_model"] = None
            result["paraphrase_prompt"] = None
            result["temperature"] = None
            result["top_p"] = None

        else:
            # Evaluation with paraphrases
            logger.info(f"Evaluating {dataset_name} with paraphrases from {gen_model}")

            # Generate or load paraphrases
            paraphrase_start = time.time()

            # Check if we should paraphrase training data
            paraphrase_train = self.config.get("paraphrase_train_data", True)
            logger.info(f"Configuration: paraphrase_train_data = {paraphrase_train}")

            # Generate new paraphrases
            if paraphrase_train:
                train_paraphrases = self.generate_paraphrases(
                    train_texts,
                    train_labels,
                    gen_model,
                    "ollama",
                    seed,
                    dataset_name,
                    split="train",
                )
            else:
                train_paraphrases = train_texts  # Use original training data

            test_paraphrases = self.generate_paraphrases(
                test_texts,
                test_labels,
                gen_model,
                "ollama",
                seed,
                dataset_name,
                split=split,
            )

            paraphrase_time = time.time() - paraphrase_start

            # Evaluate both original and paraphrased
            embedding_start = time.time()

            # Evaluate original data
            print("=== Evaluating Original Data ===")
            original_results = self.evaluator(
                embedding_model,
                train_sentences=train_texts,
                train_labels=train_labels,
                test_sentences=test_texts,
                test_labels=test_labels,
            )

            # Evaluate paraphrased data
            print("=== Evaluating Paraphrased Data ===")
            paraphrased_results = self.evaluator(
                embedding_model,
                train_sentences=train_paraphrases,
                train_labels=train_labels,
                test_sentences=test_paraphrases,
                test_labels=test_labels,
            )

            embedding_time = time.time() - embedding_start

            # Combine results
            combined_results = {}
            for key, value in original_results.items():
                combined_results[f"original_{key}"] = value
            for key, value in paraphrased_results.items():
                combined_results[f"paraphrased_{key}"] = value

            # Add standardized main score columns (accuracy for classification)
            combined_results["original_main_score"] = original_results["accuracy"]
            combined_results["paraphrased_main_score"] = paraphrased_results["accuracy"]

            # Add combined results
            result.update(combined_results)

            result["gen_model"] = gen_model
            result["gen_api"] = "ollama"
            result["paraphrase_prompt"] = self.pre_prompt
            result["paraphrase_generation_time"] = paraphrase_time
            result["embedding_generation_time"] = embedding_time
            result["temperature"] = self.temperature
            result["top_p"] = self.top_p

        return result

    def run_full_evaluation(
        self, embedding_models: List[Any], model_names: List[str], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run full evaluation across all classification datasets and models.

        Args:
            embedding_models: List of embedding models to evaluate
            model_names: List of model names corresponding to embedding_models

        Returns:
            List of evaluation results
        """
        results = []

        n_runs = self.config.get("n_runs", 1)
        seed = self.config.get("seed", 1337)

        for dataset_name in self.datasets:
            logger.info(f"\nEvaluating dataset: {dataset_name}")

            for model_idx, emb_model in enumerate(embedding_models):
                if not self.gen_model:
                    # Evaluate without paraphrases
                    logger.info(
                        f"Evaluating {dataset_name} with {model_names[model_idx]} "
                        f"(no paraphrases)"
                    )

                    result = self.evaluate_dataset(
                        dataset_name=dataset_name,
                        embedding_model=emb_model,
                        model_name=model_names[model_idx],
                        gen_model=None,
                        run=0,
                        seed=seed,
                        **kwargs,
                    )
                    results.append(result)

                    # Display main score (accuracy)
                    if "accuracy" in result:
                        print(f"  Main Score (accuracy): {result['accuracy']:.4f}")
                else:
                    # Evaluate with generative model
                    gen_model = self.gen_model
                    for run in range(n_runs):
                        run_seed = self.get_seed_for_run(run)
                        logger.info(
                            f"Evaluating {dataset_name} with "
                            f"{model_names[model_idx]} and {gen_model} (ollama) "
                            f"(run {run + 1}/{n_runs})"
                        )

                        result = self.evaluate_dataset(
                            dataset_name=dataset_name,
                            embedding_model=emb_model,
                            model_name=model_names[model_idx],
                            gen_model=gen_model,
                            run=run,
                            seed=run_seed,
                            **kwargs,
                        )
                        results.append(result)

                        # Display main scores
                        if (
                            "original_main_score" in result
                            and "paraphrased_main_score" in result
                        ):
                            print(
                                f"  Main Score - Original: {result['original_main_score']:.4f}, Paraphrased: {result['paraphrased_main_score']:.4f}"
                            )

        return results

    def load_hf_data(
        self, dataset_name: str, split: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load classification dataset from Hugging Face.

        Args:
            dataset_name: Name of the dataset
            split: Optional split to load. If None, loads both train and test splits

        Returns:
            Dictionary with 'train' and/or 'test' keys containing data
        """
        # Use self.split from config if split not provided
        eval_split = split or self.split

        # Prepend "mteb/" to dataset name
        hf_dataset_path = f"mteb/{dataset_name}"

        logger.info(f"Loading dataset {dataset_name} from {hf_dataset_path}")

        # Try loading without config first
        try:
            dataset = load_dataset(hf_dataset_path)
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
            logger.info(f"Loading {dataset_name} with language configs: {lang_configs}")
            datasets = []
            for config in lang_configs:
                try:
                    ds = load_dataset(hf_dataset_path, config)
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
            # We need to concatenate train and test splits separately
            from datasets import concatenate_datasets

            # Get the first dataset as base
            dataset = {}
            for split_name in datasets[0].keys():
                split_datasets = [ds[split_name] for ds in datasets if split_name in ds]
                if split_datasets:
                    dataset[split_name] = concatenate_datasets(split_datasets)

            logger.info(
                f"Concatenated {len(datasets)} language configs for {dataset_name}"
            )

        result = {}

        # Load train split for classification if dataset has one (needed for training)
        # Get train_split from metadata
        metadata = self.dataset_metadata.get(self.get_canonical_name(dataset_name))
        train_split_name = (
            metadata.train_split
            if metadata and hasattr(metadata, "train_split")
            else "train"
        )

        if train_split_name and train_split_name in dataset:
            train_data = dataset[train_split_name]

            # Handle different text field names
            text_field = "text" if "text" in train_data.column_names else "sentence"
            if text_field not in train_data.column_names:
                # Try other common field names
                for field in ["sentence", "premise", "question", "query"]:
                    if field in train_data.column_names:
                        text_field = field
                        break

            train_texts = list(train_data[text_field])
            train_labels = np.array(train_data["label"])

            result["train"] = {"texts": train_texts, "labels": train_labels}

        # Load evaluation split
        test_data = dataset[eval_split] if eval_split in dataset else dataset["test"]

        # Handle different text field names (same logic as train)
        text_field = "text" if "text" in test_data.column_names else "sentence"
        if text_field not in test_data.column_names:
            for field in ["sentence", "premise", "question", "query"]:
                if field in test_data.column_names:
                    text_field = field
                    break

        test_texts = list(test_data[text_field])
        test_labels = np.array(test_data["label"])

        result[eval_split] = {"texts": test_texts, "labels": test_labels}

        logger.info(
            f"Loaded {len(result['train']['texts'])} train samples and "
            f"{len(result[eval_split]['texts'])} {eval_split} samples"
        )

        # Apply downsampling if configured
        result = self.downsample_data(result)

        return result

    def generate_paraphrases(
        self,
        texts: List[str],
        labels: List[int],
        gen_model: str,
        gen_api: str,
        seed: int = 1337,
        dataset_name: str | None = None,
        split: str | None = None,
    ) -> List[str]:
        """
        Generate paraphrases for a list of texts using parent class infrastructure.

        Args:
            texts: List of texts to paraphrase
            labels: List of corresponding labels for the texts
            gen_model: Generative model name
            gen_api: API name for the generative model (hardcoded to "ollama")
            seed: Random seed for reproducibility
            dataset_name: Name of the dataset (for consistent saving)
            split: Data split name (train/test) for correct file path

        Returns:
            List of paraphrased texts
        """
        # For classification, we paraphrase each text individually
        # Use parent method with explicit seed
        paraphrased_texts = self._generate_paraphrases(
            texts=texts,
            gen_model_name=gen_model,
            gen_api_name="ollama",
            dataset_name=dataset_name,
            seed=seed,
        )

        # Save paraphrases if requested
        if self.save_paraphrases:
            seeds_used = [seed] * len(texts)
            self._save_paraphrases(
                original_texts=texts,
                paraphrased_texts=paraphrased_texts,
                labels=labels,
                split=split,
                gen_model_name=gen_model,
                gen_api_name="ollama",
                dataset_name=dataset_name,
                seeds_used=seeds_used,
            )

        return paraphrased_texts

    def evaluate(
        self,
        dataset_name: str,
        embedding_model: Any,
        emb_model_name: str | None = None,
        data: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate the embedding model on the specified dataset.

        This method is required by BaseTask but we handle evaluation
        differently in evaluate_dataset.

        Args:
            dataset_name: Name of the dataset to evaluate on
            embedding_model: The embedding model to evaluate
            emb_model_name: Name of the embedding model (optional)
            data: Dataset data (not used in our implementation)
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation results
        """
        # We handle evaluation in evaluate_dataset method
        # Use provided model name or fallback to getattr
        if emb_model_name is None:
            emb_model_name = getattr(embedding_model, "model_name", "unknown_model")

        return self.evaluate_dataset(
            dataset_name=dataset_name,
            embedding_model=embedding_model,
            model_name=emb_model_name,
            gen_model=kwargs.get("gen_model"),
            run=kwargs.get("run", 0),
            seed=kwargs.get("seed", 1337),
            **kwargs,
        )

    def _save_paraphrases(
        self,
        original_texts: List[str],
        paraphrased_texts: List[str],
        labels: List[str],
        split: str,
        gen_model_name: str,
        gen_api_name: str,
        dataset_name: str,
        seeds_used: Optional[List[Optional[int]]] = None,
        split_override: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ) -> None:
        """
        Save generated paraphrases to disk with classification-specific format.

        Args:
            original_texts: Original texts
            paraphrased_texts: Paraphrased texts
            labels: Classification labels
            split: Data split (train/test)
            gen_model_name: Name of the generative model
            gen_api_name: API name for the generative model (hardcoded to "ollama")
            dataset_name: Name of the dataset being processed
            seeds_used: Optional list of seeds used for each paraphrase
            split_override: Override the split name used in the file path
            languages: Optional list of language codes for each text
        """

        from ..utils import save_df

        # Create DataFrame with classification-specific column names
        paraphrase_df = pd.DataFrame(
            {
                "original_text": original_texts,
                "paraphrase_text": paraphrased_texts,
                "label": labels,
                "split": split,
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
        effective_split = split_override if split_override is not None else split

        # Use canonical name for path, add multilingual suffix if needed (only for STS17/STS22)
        dataset_name_for_path = self.get_canonical_name(dataset_name)
        if should_use_multilingual_suffix(dataset_name_for_path, self.config):
            dataset_name_for_path = f"{dataset_name_for_path}_multilingual"

        para_output_dir = os.path.join(
            "data",
            "paraphrases",
            "classification",
            dataset_name_for_path,
            effective_split,
            re.sub(r":", "-", gen_model_name),
        )

        os.makedirs(para_output_dir, exist_ok=True)

        # Create filename with timestamp and split
        if should_use_multilingual_suffix(dataset_name, self.config):
            para_filename = f"{self.timestamp}_{dataset_name}_{gen_model_name}_paraphrases_multilingual_{split}"
        else:
            para_filename = (
                f"{self.timestamp}_{dataset_name}_{gen_model_name}_paraphrases_{split}"
            )

        # Add seed and run number using the seed from metadata
        current_seed = (
            seeds_used[0]
            if seeds_used and len(seeds_used) > 0
            else self.seed_paraphrasing
        )
        if self.n_runs > 1:
            # Determine run number from seed
            run_offset = self.get_run_index_for_seed(current_seed)
            run_number = run_offset + 1
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

        print(f"Saved classification paraphrases ({split}) to {output_file}")
