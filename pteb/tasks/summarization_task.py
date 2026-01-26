"""
Summarization task implementation for PTEB.

This module implements the SummarizationTask class that handles summarization
dataset loading, paraphrase generation/loading, and evaluation with correlation metrics.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..evaluators.summarization_evaluator import SummarizationEvaluator
from ..utils import save_df, should_use_multilingual_suffix
from .paraphrase_task import ParaphraseTask


class SummarizationTask(ParaphraseTask):
    """
    Summarization task with paraphrase-enhanced evaluation.

    This class handles:
    - Loading summarization datasets from HuggingFace Hub
    - Generating or loading paraphrases of summaries
    - Evaluating embedding models on original and paraphrased data
    - Computing correlations between embedding similarities and human scores
    """

    def __init__(self, datasets: List[str], config: Dict[str, Any], **kwargs):
        """
        Initialize the Summarization task.

        Args:
            datasets: List of summarization dataset names to evaluate on
            config: Configuration dictionary from CLI arguments
            **kwargs: Additional parameters
        """

        super().__init__(
            task_name="Summarization",
            task_description="Evaluate machine summaries against human summaries and assessments",
            datasets=datasets,
            config=config,
            **kwargs,
        )

        # Initialize evaluator
        self.evaluator = SummarizationEvaluator(
            similarity_metrics=config.get("similarity_metrics", ["cosine"]),
            task_name="summarization",
            seed=self.seed_general,
        )

    def downsample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downsample summarization samples."""
        downsample = self.config.get("downsample")
        if downsample is None or downsample >= 1.0:
            return df

        n_sample = max(1, int(len(df) * downsample))
        df_sampled = df.sample(n=n_sample, random_state=self.seed_general).reset_index(drop=True)

        print(f"Downsampled to {n_sample} samples ({downsample*100:.1f}%)")
        return df_sampled

    def load_hf_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Load and return the specified summarization dataset from HuggingFace Hub.

        Args:
            dataset_name: Name of the summarization dataset to load

        Returns:
            DataFrame containing the loaded dataset with required columns

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
            hf_dataset = load_dataset(hf_dataset_name, split=self.split)

            # Convert to pandas DataFrame
            df = hf_dataset.to_pandas()

            # Ensure basic required columns exist
            required_cols = ["human_summaries", "machine_summaries"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns: {missing_cols}. Available: {list(df.columns)}"
                )

            # Handle different score column formats
            score_cols = [
                "scores",
                "relevance",
                "coherence",
                "fluency",
                "consistency",
            ]
            available_score_cols = [col for col in score_cols if col in df.columns]

            if not available_score_cols:
                raise ValueError(
                    f"No score columns found. Expected one of: {score_cols}. Available: {list(df.columns)}"
                )

            # If we don't have a unified 'scores' column, create one from available dimensions
            if "scores" not in df.columns:
                if "relevance" in df.columns:
                    # Use relevance as the primary metric (most important for summarization)
                    df["scores"] = df["relevance"]
                    print("Using 'relevance' scores as primary metric for evaluation")
                else:
                    # Use the first available score column
                    primary_score_col = available_score_cols[0]
                    df["scores"] = df[primary_score_col]
                    print(
                        f"Using '{primary_score_col}' scores as primary metric for evaluation"
                    )

            # Convert numpy arrays to lists (HuggingFace datasets conversion creates numpy arrays)
            for col in ["human_summaries", "machine_summaries", "scores"]:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: x.tolist() if hasattr(x, "tolist") else x
                    )

            # Validate data structure
            self._validate_dataset_structure(df, dataset_name)

            print(f"Loaded {len(df)} samples from {dataset_name}")

            # Apply downsampling if configured
            df = self.downsample_data(df)

            return df

        except Exception as e:
            print(f"Error loading dataset {hf_dataset_name}: {e}")
            raise

    def evaluate(
        self,
        dataset_name: str,
        embedding_model: Any,
        emb_model_name: str,
        gen_model_name: Optional[str] = None,
        gen_api_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate the embedding model on summarization data with paraphrases.

        Args:
            dataset_name: Name of the dataset being evaluated
            embedding_model: The embedding model to evaluate
            emb_model_name: Name of the embedding model
            data: DataFrame with the dataset (if None, will load from dataset_name)
            gen_model_name: Name of generative model for paraphrases
            gen_api_name: API name for generative model
            pregenerated_paraphrases: Pre-generated paraphrase data
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation results
        """
        return self.evaluate_online(
            dataset_name,
            embedding_model,
            emb_model_name,
            gen_model_name,
            gen_api_name,
            **kwargs,
        )

    def evaluate_online(
        self,
        dataset_name: str,
        embedding_model: Any,
        emb_model_name: str,
        gen_model_name: str,
        gen_api_name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate with online paraphrase generation."""
        print(f"Evaluating {dataset_name} with paraphrases from {gen_model_name}")

        data = self.load_hf_data(dataset_name)

        human_summaries = data["human_summaries"].tolist()
        machine_summaries = data["machine_summaries"].tolist()
        scores = data["scores"].tolist()

        # Generate new paraphrases (automatically saved by parent class)
        print(f"Generating paraphrases for {dataset_name} with {gen_model_name}...")
        paraphrased_human, paraphrased_machine = self.generate_paraphrases(
            human_summaries,
            machine_summaries,
            gen_model_name,
            gen_api_name,
            kwargs.get("seed", self.seed_paraphrasing),
            dataset_name,
            gold_scores=scores,
        )

        # Evaluate original summaries
        print("=== Evaluating Original Summaries ===")
        original_result = self.evaluator(
            embedding_model,
            human_summaries=human_summaries,
            machine_summaries=machine_summaries,
            gold_scores=scores,
            texts=None,  # Texts not needed for summarization evaluation
            phase="Original",
        )

        # Evaluate paraphrased summaries
        print("=== Evaluating Paraphrased Summaries ===")
        paraphrased_result = self.evaluator(
            embedding_model,
            human_summaries=paraphrased_human,
            machine_summaries=paraphrased_machine,
            gold_scores=scores,
            texts=None,  # Texts not needed for summarization evaluation
            phase="Paraphrased",
        )

        # Combine results
        result = self._combine_results(original_result, paraphrased_result)

        # Add metadata
        metadata = self._get_metadata(
            emb_model_name, embedding_model, gen_model_name, gen_api_name, **kwargs
        )
        metadata["dataset"] = self.get_canonical_name(dataset_name)
        metadata["n_samples"] = len(data) if data is not None else len(human_summaries)
        result.update(metadata)

        return result

    def _get_metadata(
        self,
        emb_model_name: str,
        embedding_model: Any,
        gen_model_name: str | None = None,
        gen_api_name: str | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Get metadata for evaluation results."""
        metadata = {
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
            "gen_api": gen_api_name,
            "paraphrase_prompt": self.pre_prompt if gen_model_name else None,
            "temperature": self.temperature if gen_model_name else None,
            "top_p": self.top_p if gen_model_name else None,
            "task": "Summarization",
            "timestamp": self.timestamp,
            "embedding_generation_time": kwargs.get("embedding_generation_time", None),
        }

        # Add run information if seed is provided (for multiple runs)
        if "seed" in kwargs:
            run_seed = kwargs["seed"]
            run_offset = self.get_run_index_for_seed(run_seed)
            run_number = run_offset + 1
            metadata["run"] = run_number
            metadata["seed"] = run_seed
        else:
            metadata["run"] = kwargs.get("run", 0)

        return metadata

    def _combine_results(
        self, original_result: Dict[str, Any], paraphrased_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine original and paraphrased evaluation results."""
        result = {}
        for key, value in original_result.items():
            result[f"original_{key}"] = value
        for key, value in paraphrased_result.items():
            result[f"paraphrased_{key}"] = value

        # Add standardized main score columns (spearman for summarization)
        result["original_main_score"] = original_result["spearman"]
        result["paraphrased_main_score"] = paraphrased_result["spearman"]

        return result

    def _extract_main_score(
        self, evaluation_result: Dict[str, Any], prefix: str
    ) -> float:
        """Extract main score (spearman) from evaluation result."""
        return evaluation_result.get("spearman")

    def _validate_dataset_structure(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Validate that the dataset has the required structure.

        Args:
            df: Dataset DataFrame
            dataset_name: Name of the dataset

        Raises:
            ValueError: If required columns are missing or data is malformed
        """
        # Validate data types and structure for a few sample rows
        sample_size = min(3, len(df))  # Check first 3 rows or all if less than 3

        for i in range(sample_size):
            row = df.iloc[i]
            try:
                # Check that the columns contain lists
                if not isinstance(row["human_summaries"], list):
                    raise ValueError(
                        f"Row {i}: human_summaries must be a list, got {type(row['human_summaries'])}"
                    )
                if not isinstance(row["machine_summaries"], list):
                    raise ValueError(
                        f"Row {i}: machine_summaries must be a list, got {type(row['machine_summaries'])}"
                    )
                if not isinstance(row["scores"], list):
                    raise ValueError(
                        f"Row {i}: scores must be a list, got {type(row['scores'])}"
                    )

                # Check alignment between machine summaries and scores
                if len(row["machine_summaries"]) != len(row["scores"]):
                    raise ValueError(
                        f"Row {i}: machine_summaries ({len(row['machine_summaries'])}) and scores ({len(row['scores'])}) must have same length"
                    )

                # Check that we have some human summaries
                if len(row["human_summaries"]) == 0:
                    raise ValueError(f"Row {i}: must have at least one human summary")

                # Check that we have some machine summaries and scores
                if len(row["machine_summaries"]) == 0:
                    raise ValueError(f"Row {i}: must have at least one machine summary")

                print(
                    f"✓ Row {i} validation passed: {len(row['human_summaries'])} human summaries, {len(row['machine_summaries'])} machine summaries"
                )

            except (KeyError, TypeError) as e:
                raise ValueError(f"Row {i}: invalid data structure - {e}")

    def run_full_evaluation(
        self, embedding_models: List[Any], model_names: List[str], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run full evaluation across all datasets and embedding models.

        This method handles generative models properly for paraphrase generation.

        Args:
            embedding_models: List of embedding models to evaluate
            model_names: List of model names corresponding to embedding_models

        Returns:
            List of evaluation results
        """
        results = []

        n_runs = self.config.get("n_runs", 1)
        # Use seed_paraphrasing for paraphrase operations
        seed_paraphrasing = self.config.get(
            "seed_paraphrasing", self.config.get("seed", 1337)
        )

        for dataset_name in self.datasets:
            print(f"\n{'=' * 40}")
            print(f"Dataset: {dataset_name}")
            print(f"{'=' * 40}")

            for model_idx, emb_model in enumerate(embedding_models):
                model_name = model_names[model_idx]
                if not self.gen_model:
                    # Evaluate without paraphrases
                    print(f"Evaluating with model: {model_name}")

                    result = self.evaluate(
                        dataset_name=dataset_name,
                        embedding_model=emb_model,
                        emb_model_name=model_name,
                        gen_model_name=None,
                        gen_api_name=None,
                    )
                    results.append(result)

                    # Display main score (spearman)
                    if "spearman" in result:
                        print(f"  Main Score (spearman): {result['spearman']:.4f}")
                else:
                    # Evaluate with generative model
                    gen_model = self.gen_model
                    gen_api = self.gen_model_api
                    for run in range(n_runs):
                        run_seed = self.get_seed_for_run(run)
                        print(f"Evaluating with model: {model_name}")
                        print(f"  Using generative model: {gen_model} ({gen_api})")

                        result = self.evaluate(
                            dataset_name=dataset_name,
                            embedding_model=emb_model,
                            emb_model_name=model_name,
                            gen_model_name=gen_model,
                            gen_api_name=gen_api,
                            seed=run_seed,
                            run=run,
                        )
                        results.append(result)

                        # Display main scores
                        if (
                            "original_main_score" in result
                            and "paraphrased_main_score" in result
                        ):
                            print(
                                f"    Main Score - Original: {result['original_main_score']:.4f}, Paraphrased: {result['paraphrased_main_score']:.4f}"
                            )

                print(f"Completed in {result.get('runtime', 0):.2f} seconds")

        return results

    def _save_paraphrases(
        self,
        human_summaries_nested: List[List[str]],
        machine_summaries_nested: List[List[str]],
        paraphrased_human_nested: List[List[str]],
        paraphrased_machine_nested: List[List[str]],
        gold_scores: List[List[float]],
        gen_model_name: str,
        gen_api_name: str,
        dataset_name: str,
        seed: int = 1337,
        split_override: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ) -> None:
        """
        Save generated paraphrases to disk preserving nested structure for summarization.

        Args:
            human_summaries_nested: Original human summaries (nested: samples -> summaries)
            machine_summaries_nested: Original machine summaries (nested: samples -> summaries)
            paraphrased_human_nested: Paraphrased human summaries (nested: samples -> summaries)
            paraphrased_machine_nested: Paraphrased machine summaries (nested: samples -> summaries)
            gold_scores: Gold scores for machine summaries (nested: samples -> scores)
            gen_model_name: Name of the generative model
            gen_api_name: API name for the generative model
            dataset_name: Name of the dataset being processed
            seed: Random seed used for generation
            split_override: Override the split name used in the file path
            languages: Optional list of language codes for each sample
        """

        # Determine output directory - use split_override if provided, otherwise use self.split
        effective_split = split_override if split_override is not None else self.split

        # Use canonical name for path, add multilingual suffix if needed (only for STS17/STS22)
        dataset_name_for_path = self.get_canonical_name(dataset_name)
        if should_use_multilingual_suffix(dataset_name_for_path, self.config):
            dataset_name_for_path = f"{dataset_name_for_path}_multilingual"

        para_output_dir = os.path.join(
            "data",
            "paraphrases",
            "summarization",
            dataset_name_for_path,
            effective_split,
            re.sub(r"[:/]", "-", gen_model_name),
        )

        os.makedirs(para_output_dir, exist_ok=True)

        # Create filename - always use JSONL for summarization to preserve structure
        if should_use_multilingual_suffix(dataset_name, self.config):
            para_filename = f"{self.timestamp}_{dataset_name}_{gen_model_name}_paraphrases_multilingual_{effective_split}"
        else:
            para_filename = f"{self.timestamp}_{dataset_name}_{gen_model_name}_paraphrases_{effective_split}"

        # Add seed and run number for new filename pattern
        if self.n_runs > 1:
            # Determine run number from seed if provided
            if seed != self.seed_paraphrasing:
                run_offset = self.get_run_index_for_seed(seed)
                run_number = run_offset + 1
            else:
                run_number = 1  # Default to run1 if seed matches base seed
            para_filename += f"_seed-{seed:04d}_run-{run_number}"
        else:
            # Single run case
            para_filename += f"_seed-{seed:04d}_run-1"

        para_filename = re.sub(
            r":", "-", para_filename
        )  # Sanitize model name in filename
        para_filename += ".jsonl"

        # Get default language from metadata if not provided
        if not languages:
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
            languages = [default_lang] * len(human_summaries_nested)

        # Prepare data for saving using the centralized save_df function
        sample_data_list = []
        for i in range(len(human_summaries_nested)):
            sample_data = {
                "sample_id": i,
                "original_human_summaries": human_summaries_nested[i],
                "original_machine_summaries": machine_summaries_nested[i],
                "paraphrase_human_summaries": paraphrased_human_nested[i],
                "paraphrase_machine_summaries": paraphrased_machine_nested[i],
                "scores": gold_scores[i],
                "language": languages[i],
            }

            sample_data_list.append(sample_data)

        # Convert to DataFrame for save_df function
        paraphrase_df = pd.DataFrame(sample_data_list)

        # Remove the .jsonl extension from para_filename since save_df adds it
        para_filename_base = para_filename.replace(".jsonl", "")
        para_output_path = os.path.join(para_output_dir, para_filename_base)

        # Save using the centralized function with overwrite protection
        save_df(paraphrase_df, para_output_path, file_format="jsonl")

        print("Summarization paraphrases saved (JSONL format with nested structure)")

    def generate_paraphrases(
        self,
        human_summaries: List[List[str]],
        machine_summaries: List[List[str]],
        gen_model: str,
        gen_api: str,
        seed: int = 1337,
        dataset_name: str | None = None,
        gold_scores: list[list[float]] | None = None,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Generate paraphrases for human and machine summaries.

        Args:
            human_summaries: List of lists of human summary texts
            machine_summaries: List of lists of machine summary texts
            gen_model: Name of generative model
            gen_api: API name for generative model
            seed: Random seed for reproducibility
            dataset_name: Name of dataset (for saving)
            gold_scores: Gold scores for machine summaries (for saving)

        Returns:
            Tuple of (paraphrased_human_summaries, paraphrased_machine_summaries)
        """
        # Flatten the nested lists for paraphrase generation
        # Keep track of lengths to reshape later
        human_lengths = [len(summaries) for summaries in human_summaries]
        machine_lengths = [len(summaries) for summaries in machine_summaries]

        flat_human_summaries = [
            summary for summaries in human_summaries for summary in summaries
        ]
        flat_machine_summaries = [
            summary for summaries in machine_summaries for summary in summaries
        ]

        # Generate paraphrases for flattened lists
        paraphrased_human_flat = self._generate_paraphrases(
            texts=flat_human_summaries,
            gen_model_name=gen_model,
            gen_api_name=gen_api,
            dataset_name=dataset_name,
        )

        paraphrased_machine_flat = self._generate_paraphrases(
            texts=flat_machine_summaries,
            gen_model_name=gen_model,
            gen_api_name=gen_api,
            dataset_name=dataset_name,
        )

        # Reshape back to nested lists
        paraphrased_human = []
        paraphrased_machine = []

        human_idx = 0
        for length in human_lengths:
            paraphrased_human.append(
                paraphrased_human_flat[human_idx : human_idx + length]
            )
            human_idx += length

        machine_idx = 0
        for length in machine_lengths:
            paraphrased_machine.append(
                paraphrased_machine_flat[machine_idx : machine_idx + length]
            )
            machine_idx += length

        # Save paraphrases with custom format (only if saving is enabled)
        if self.save_paraphrases:
            # Use provided gold scores or create placeholder if not available
            if gold_scores is None:
                gold_scores = [
                    [0.0] * len(machine_summaries[i])
                    for i in range(len(machine_summaries))
                ]

            self._save_paraphrases(
                human_summaries_nested=human_summaries,
                machine_summaries_nested=machine_summaries,
                paraphrased_human_nested=paraphrased_human,
                paraphrased_machine_nested=paraphrased_machine,
                gold_scores=gold_scores,
                gen_model_name=gen_model,
                gen_api_name=gen_api,
                dataset_name=dataset_name,
                seed=seed,
            )

        return paraphrased_human, paraphrased_machine

    def load_disk_data(
        self,
        dataset_name: str,
        split: str,
        gen_model: str,
        task_type: str = "summarization",
        original_human_summaries: list[list[str]] | None = None,
        original_machine_summaries: list[list[str]] | None = None,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Load pre-generated paraphrases from disk with summarization-specific format.

        Args:
            dataset_name: Name of the dataset
            split: Data split (e.g., "test")
            gen_model: Name of generative model used
            task_type: Task type directory name
            original_human_summaries: Original nested human summaries structure for reshaping
            original_machine_summaries: Original nested machine summaries structure for reshaping

        Returns:
            Tuple of (paraphrased_human_summaries, paraphrased_machine_summaries) with nested structure
        """
        import glob

        # Construct paraphrase directory path using parent class structure
        para_dir = os.path.join(
            "data",
            "paraphrases",
            task_type,
            dataset_name,
            split,
            gen_model.replace(":", "-").replace("/", "-"),
        )

        if not os.path.exists(para_dir):
            raise FileNotFoundError(
                f"Paraphrase directory not found: {para_dir}. "
                f"Please generate paraphrases first using online mode."
            )

        # Find paraphrase files - prioritize JSONL for summarization
        jsonl_pattern = os.path.join(para_dir, "*.jsonl")
        jsonl_files = glob.glob(jsonl_pattern)

        if jsonl_files:
            # Use JSONL file (new format with nested structure)
            latest_file = max(jsonl_files, key=os.path.getctime)
            print(
                f"Loading summarization paraphrases from {latest_file} (JSONL format)"
            )

            paraphrased_human_nested = []
            paraphrased_machine_nested = []

            with open(latest_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        sample_data = json.loads(line.strip())
                        paraphrased_human_nested.append(
                            sample_data["paraphrase_human_summaries"]
                        )
                        paraphrased_machine_nested.append(
                            sample_data["paraphrase_machine_summaries"]
                        )

            print(
                f"Loaded {len(paraphrased_human_nested)} samples with nested structure"
            )
            return paraphrased_human_nested, paraphrased_machine_nested

        else:
            # Fallback to legacy formats (CSV, Excel) - flatten and reshape
            legacy_patterns = [
                os.path.join(para_dir, "*.csv"),
                os.path.join(para_dir, "*.xlsx"),
                os.path.join(para_dir, "*.xls"),
            ]

            files = []
            for pattern in legacy_patterns:
                files.extend(glob.glob(pattern))

            if not files:
                raise FileNotFoundError(f"No paraphrase files found in {para_dir}")

            # Use the most recent file
            latest_file = max(files, key=os.path.getctime)
            print(
                f"Loading summarization paraphrases from {latest_file} (legacy format)"
            )

            # Load data based on file extension
            if latest_file.endswith(".csv"):
                df = pd.read_csv(latest_file)
            elif latest_file.endswith((".xlsx", ".xls")):
                df = pd.read_excel(latest_file)
            else:
                raise ValueError(f"Unsupported file format: {latest_file}")

            # Summarization-specific columns
            if (
                "paraphrase_human_summaries" in df.columns
                and "paraphrase_machine_summaries" in df.columns
            ):
                print("Found legacy summarization format with dedicated columns")
                human_paraphrases = df["paraphrase_human_summaries"].tolist()
                machine_paraphrases = df["paraphrase_machine_summaries"].tolist()

            # STS-style format: sentence1=machine, sentence2=human
            elif (
                "paraphrase_sentence1" in df.columns
                and "paraphrase_sentence2" in df.columns
            ):
                print("Found STS format, using sentence1=machine, sentence2=human")
                machine_paraphrases = df["paraphrase_sentence1"].tolist()
                human_paraphrases = df["paraphrase_sentence2"].tolist()

            # Single column format: split evenly between human and machine
            elif "paraphrase_sentence1" in df.columns:
                print("Warning: Found only paraphrase_sentence1, splitting evenly")
                paraphrases = df["paraphrase_sentence1"].tolist()
                mid_point = len(paraphrases) // 2
                human_paraphrases = paraphrases[:mid_point]
                machine_paraphrases = paraphrases[mid_point:]

            else:
                raise ValueError(
                    f"Could not find paraphrase columns in file {latest_file}. "
                    f"Expected 'paraphrase_human_summaries' and 'paraphrase_machine_summaries' "
                    f"or legacy 'paraphrase_sentence1'/'paraphrase_sentence2' columns. "
                    f"Found columns: {list(df.columns)}"
                )

            print(
                f"Loaded {len(human_paraphrases)} human and {len(machine_paraphrases)} machine paraphrases (flat)"
            )

            # Reshape flat lists back to nested structure if original structure is provided
            if (
                original_human_summaries is not None
                and original_machine_summaries is not None
            ):
                print("Reshaping paraphrases to match original nested structure...")

                # Get lengths for each sample
                human_lengths = [
                    len(summaries) for summaries in original_human_summaries
                ]
                machine_lengths = [
                    len(summaries) for summaries in original_machine_summaries
                ]

                # Reshape human paraphrases
                reshaped_human = []
                human_idx = 0
                for length in human_lengths:
                    sample_paraphrases = human_paraphrases[
                        human_idx : human_idx + length
                    ]
                    reshaped_human.append(sample_paraphrases)
                    human_idx += length

                # Reshape machine paraphrases
                reshaped_machine = []
                machine_idx = 0
                for length in machine_lengths:
                    sample_paraphrases = machine_paraphrases[
                        machine_idx : machine_idx + length
                    ]
                    reshaped_machine.append(sample_paraphrases)
                    machine_idx += length

                print(
                    f"Reshaped to {len(reshaped_human)} samples with human summaries and {len(reshaped_machine)} samples with machine summaries"
                )
                return reshaped_human, reshaped_machine
            else:
                # Return as flat lists (backward compatibility)
                return human_paraphrases, machine_paraphrases

    def __str__(self) -> str:
        """String representation of the task."""
        return f"SummarizationTask(datasets={self.datasets})"
