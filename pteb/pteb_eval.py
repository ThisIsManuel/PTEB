"""
Main evaluator script for PTEB.

This is the main entry point that provides a clean interface for running
evaluations using the class-based architecture.
"""

import argparse
import datetime
import gc
import os
import sys
import time
import warnings
from typing import Any

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from .dataset_metadata import get_dataset_metadata
from .tasks.classification_task import ClassificationTask
from .tasks.clustering_task import ClusteringTask
from .tasks.pair_classification_task import PairClassificationTask
from .tasks.reranking_task import RerankingTask
from .tasks.retrieval_task import RetrievalTask
from .tasks.sts_task import STSTask
from .tasks.summarization_task import SummarizationTask
from .utils import play_completion_sound


def get_task_type_from_dataset(dataset_name: str) -> str:
    """
    Automatically determine task type from dataset name using metadata.

    Args:
        dataset_name: Name of the dataset (case-insensitive)

    Returns:
        Task type string (e.g., "sts", "pair_classification", "clustering")

    Raises:
        ValueError: If dataset not found in metadata registry
    """
    try:
        metadata = get_dataset_metadata(dataset_name)
        return metadata.task_type
    except ValueError as e:
        # Provide helpful error message with available datasets
        raise ValueError(
            f"Dataset '{dataset_name}' not found in metadata registry. "
            f"Please check the dataset name spelling or add it to pteb/dataset_metadata/. "
            f"Original error: {e}"
        ) from e


def create_embedding_model(
    model_name: str, config: dict[str, Any]
) -> SentenceTransformer:
    """
    Create a single embedding model instance from configuration with memory optimizations.

    Args:
        model_name: Name of the model to load
        config: Configuration dictionary

    Returns:
        SentenceTransformer instance
    """
    # Convert string dtype to torch dtype if needed
    st_dtype = config.get("st_dtype", "torch.float16")
    if isinstance(st_dtype, str):
        dtype_map = {
            "torch.float16": torch.float16,
            "torch.float32": torch.float32,
            "torch.bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        st_dtype = dtype_map.get(st_dtype, torch.float16)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare model kwargs
    model_kwargs = {
        "torch_dtype": st_dtype,
    }

    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs=model_kwargs,
        trust_remote_code=True,
    )

    # Use model's default max_seq_length
    print(f"Loaded SentenceTransformer model: {model_name}")
    print(f"    - dtype set to {st_dtype}")
    print(f"    - Using model default max_seq_length: {model.max_seq_length}")

    # Store model name as attribute for task-specific prompt handling
    setattr(model, "model_name", model_name)

    return model


def clear_gpu_memory(aggressive: bool = False) -> None:
    """
    Clear GPU memory and run garbage collection with optional aggressive cleaning.

    This function helps free up GPU memory between model loads to prevent
    out-of-memory errors when loading large models sequentially.

    Args:
        aggressive: If True, performs multiple rounds of cleanup and synchronization
    """
    for _ in range(3 if aggressive else 1):
        gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        # Synchronize all streams before clearing cache
        torch.cuda.synchronize()

        # Clear cache multiple times for aggressive cleanup
        for _ in range(2 if aggressive else 1):
            torch.cuda.empty_cache()

        # Reset peak memory stats for better monitoring
        torch.cuda.reset_peak_memory_stats()

        print(
            f"GPU memory cleared{' (aggressive)' if aggressive else ''}. "
            f"Current usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB reserved"
        )


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    # Check for required datasets
    if not config.get("datasets"):
        raise ValueError("No datasets specified in configuration")

    # Correct task names
    task_mapping = {
        "pair-classification": "pair_classification",
        "pairclassification": "pair_classification",
        "summarisation": "summarization",
    }
    # Map task names to their correct names
    for task in config["datasets"]:
        if task in task_mapping:
            config["datasets"][task] = task_mapping[task]

    # tasks to remove if no datasets specified
    tasks_to_remove = []
    # Check that for all tasks at least one dataset is specified
    for task, datasets in config["datasets"].items():
        if datasets is None or not isinstance(datasets, list) or len(datasets) == 0:
            warnings.warn(f"No valid datasets specified for task '{task}'")
            tasks_to_remove.append(task)

    for task in tasks_to_remove:
        config["datasets"].pop(task)
    # Check for at least one task type
    supported_tasks = [
        "sts",
        "pair_classification",
        "clustering",
        "retrieval",
        "summarization",
        "reranking",
        "classification",
    ]
    has_task = any(task in config["datasets"] for task in supported_tasks)
    if not has_task:
        raise ValueError(f"At least one task type must be specified: {supported_tasks}")

    # Check generative model configuration
    gen_models = config.get("gen_models", [])
    gen_apis = config.get("gen_model_apis", [])

    # Evaluation can run without generative models (original data only) or with them (paraphrases)
    if gen_models and not gen_apis:
        raise ValueError(
            "Evaluation with generative models requires generative model APIs to be specified"
        )

    # Check API count matches model count for generative models (if specified)
    if (
        gen_models
        and gen_apis
        and len(gen_apis) != 1
        and len(gen_apis) != len(gen_models)
    ):
        raise ValueError(
            "Number of generative APIs must be 1 or match number of generative models"
        )


def print_task_hyperparameters(task_name: str, config: dict[str, Any]) -> None:
    """
    Print hyperparameters for a specific task.

    Args:
        task_name: Name of the task
        config: Configuration dictionary
    """
    print(f"\n{'=' * 60}")
    print(f"{task_name} Task Hyperparameters")
    print(f"{'=' * 60}")

    # General parameters
    print("General Parameters:")
    seed = config.get("seed", 1337)
    n_runs = config.get("n_runs", 1)
    if n_runs > 1:
        print(f"  Seed: {seed}-{seed + n_runs - 1} ({n_runs} runs)")
    else:
        print(f"  Seed: {seed}")

    print(f"  Embedding batch size: {config.get('emb_batch_size', 32)}")
    print(f"  Embedding API: {config.get('emb_model_api', 'sentence-transformers')}")
    print(f"  Data type: {config.get('st_dtype', 'torch.float32')}")
    print(f"  Split: {config.get('split', 'test')}")

    if config.get("gen_models"):
        print(f"  Temperature: {config.get('temperature', 0.7)}")
        print(f"  Top-p: {config.get('top_p', 1.0)}")

    # Task-specific parameters
    print(f"\n{task_name}-Specific Parameters:")

    if task_name == "STS":
        if config.get("languages"):
            print(f"  MTEB languages: {config.get('languages')}")

    elif task_name == "Classification":
        print(f"  Max iterations (LogReg): {config.get('max_iter', 100)}")
        print(f"  Paraphrase train data: {config.get('paraphrase_train_data', True)}")

    elif task_name == "Clustering":
        print(f"  Clustering batch size: {config.get('clustering_batch_size', 32)}")

    elif task_name == "Retrieval":
        if config.get("corpus_chunk_size"):
            print(f"  Corpus chunk size: {config.get('corpus_chunk_size')}")

    elif task_name == "Reranking":
        print("  Use batched encoding: True (always enabled)")

    elif task_name == "Summarization":
        print(f"  Similarity metrics: {config.get('similarity_metrics', ['cosine'])}")

    elif task_name == "PairClassification":
        print("  No task-specific parameters")

    print(f"{'=' * 60}\n")


def print_evaluation_summary(results: list[dict[str, Any]], total_time: float) -> None:
    """
    Print a summary of evaluation results.

    Args:
        results: List of evaluation results
        total_time: Total evaluation time in seconds
    """
    if not results:
        print("No results to display")
        return

    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    # Create summary table

    # Extract key metrics for display
    summary_data = []
    for result in results:
        row = {
            "Dataset": result.get("dataset", "Unknown"),
            "Task": result.get("task", "Unknown"),
            "Emb_Model": result.get("emb_model", "Unknown").replace(
                "sentence-transformers/", ""
            ),
            "Gen_Model": result.get("gen_model", "None"),
            "Samples": result.get("n_samples", "Unknown"),
        }

        # Add main metrics - unified approach for all tasks
        # For original evaluation (no generative models)
        if result.get("gen_model") is None:
            # No paraphrases - get the primary metric from original_main_score field
            # This field is set by evaluators using their get_main_score_name() method
            original_main = result.get("original_main_score", 0)
            row["Original_Main"] = f"{original_main:.2f}"
            row["Paraphrased_Main"] = "NaN"
        else:
            # For paraphrased evaluation - use standardized main_score fields
            original_main = result.get("original_main_score", 0)
            paraphrased_main = result.get("paraphrased_main_score", 0)
            row["Original_Main"] = f"{original_main:.2f}"
            row["Paraphrased_Main"] = f"{paraphrased_main:.2f}"
        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Replace long names for cleaner display
    df["Dataset"] = df["Dataset"].replace("stsbenchmark", "STSB")

    print(df.to_string(index=False))

    # Create summary table by dataset (averaged over runs)
    if len(results) > 1:
        # Create a DataFrame from the summary data for aggregation
        summary_df = pd.DataFrame(summary_data)

        # Check if we have numeric columns to aggregate
        numeric_cols = []
        if "Original_Main" in summary_df.columns:
            # Convert to numeric for aggregation (remove the string formatting)
            summary_df["Original_Main_Numeric"] = pd.to_numeric(
                summary_df["Original_Main"], errors="coerce"
            )
            numeric_cols.append("Original_Main_Numeric")
        if (
            "Paraphrased_Main" in summary_df.columns
            and summary_df["Paraphrased_Main"].iloc[0] != "NaN"
        ):
            # Only process if not all values are NaN
            non_nan_para = summary_df[summary_df["Paraphrased_Main"] != "NaN"]
            if not non_nan_para.empty:
                summary_df["Paraphrased_Main_Numeric"] = pd.to_numeric(
                    summary_df["Paraphrased_Main"], errors="coerce"
                )
                numeric_cols.append("Paraphrased_Main_Numeric")

        # Convert Samples to numeric if possible
        summary_df["Samples_Numeric"] = pd.to_numeric(
            summary_df["Samples"], errors="coerce"
        )

        # Group by task, dataset, emb_model, gen_model
        group_cols = ["Task", "Dataset", "Emb_Model", "Gen_Model"]
        grouped = summary_df.groupby(group_cols)

        # Check if we have any groups with multiple runs
        has_multi_run_groups = any(len(group) > 1 for name, group in grouped)

        if has_multi_run_groups:
            print(f"\n{'=' * 80}")
            print("SUMMARY BY DATASET (Averaged over runs)")
            print(f"{'=' * 80}")

            dataset_summary = []
            for name, group in grouped:
                if len(group) > 1:  # Only show summary for multi-run groups
                    task, dataset, emb_model, gen_model = name  # type: ignore[not-iterable]

                    summary_row = {
                        "Task": task,
                        "Dataset": dataset,
                        "Emb_Model": emb_model,
                        "Gen_Model": gen_model,
                        "Runs": len(group),
                    }

                    # Calculate averages for scores
                    original_mean = None
                    paraphrase_mean = None

                    if (
                        "Original_Main_Numeric" in group.columns
                        and group["Original_Main_Numeric"].notna().any()
                    ):
                        original_mean = group["Original_Main_Numeric"].mean()
                        # Calculate standard deviation if we have multiple values
                        if len(group) > 1:
                            original_std = group["Original_Main_Numeric"].std()
                            summary_row["Original_Score"] = (
                                f"{original_mean:.2f} (±{original_std:.2f})"
                            )
                        else:
                            summary_row["Original_Score"] = f"{original_mean:.2f}"
                    else:
                        summary_row["Original_Score"] = "NaN"

                    if (
                        "Paraphrased_Main_Numeric" in group.columns
                        and group["Paraphrased_Main_Numeric"].notna().any()
                    ):
                        paraphrase_mean = group["Paraphrased_Main_Numeric"].mean()
                        # Calculate standard deviation if we have multiple values
                        if len(group) > 1:
                            std_score = group["Paraphrased_Main_Numeric"].std()
                            summary_row["Paraphrase_Score"] = (
                                f"{paraphrase_mean:.2f} (±{std_score:.2f})"
                            )
                        else:
                            summary_row["Paraphrase_Score"] = f"{paraphrase_mean:.2f}"
                    else:
                        summary_row["Paraphrase_Score"] = "NaN"

                    # Calculate Delta (Paraphrase - Original)
                    if original_mean is not None and paraphrase_mean is not None:
                        delta = paraphrase_mean - original_mean
                        summary_row["Delta"] = f"{delta:.2f}"
                    else:
                        summary_row["Delta"] = "NaN"

                    dataset_summary.append(summary_row)

            if dataset_summary:
                summary_table_df = pd.DataFrame(dataset_summary)
                print(summary_table_df.to_string(index=False))

    # Show paraphrase prompts used
    unique_prompts = {}
    for result in results:
        gen_model = result.get("gen_model")
        prompt = result.get("paraphrase_prompt")
        if gen_model and gen_model != "None" and prompt:
            unique_prompts[gen_model] = prompt

    if unique_prompts:
        print("\n" + "=" * 60)
        print("PARAPHRASE PROMPTS USED")
        print("=" * 60)
        for model, prompt in unique_prompts.items():
            # Truncate long prompts for display
            display_prompt = prompt if len(prompt) <= 80 else prompt[:77] + "..."
            print(f"• Model: {model}")
            print(f'  Prompt: "{display_prompt}"')

    print(f"\nTotal evaluation time: {total_time:.2f} seconds")
    print(f"Total evaluations: {len(results)}")


def save_evaluation_results(
    results: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    """
    Save evaluation results to file.

    Args:
        results: List of evaluation results
        config: Configuration dictionary
    """
    if not results:
        print("No results to save")
        return

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Metric scaling handled by BaseEvaluator._apply_percentage_scaling()

    # Remove unwanted columns
    columns_to_remove = ["individual_eval_time", "runtime"]
    for col in columns_to_remove:
        if col in results_df.columns:
            results_df = results_df.drop(columns=[col])

    # Reorder columns according to specified order
    desired_column_order = [
        "timestamp",
        "run",
        "gen_api",
        "gen_model",
        "paraphrase_prompt",
        "temperature",
        "top_p",
        "seed_paraphrasing",
        "seed_general",
        "emb_model_api",
        "emb_model",
        "model_max_seq_length",
        "emb_batch_size",
        "dtype",
        "task",
        "dataset",
        "n_samples",
        "paraphrase_generation_time",
        "embedding_generation_time",
        "main_metric",
        "original_main_score",
        "paraphrased_main_score",
    ]

    # Reorder columns, keeping only those that exist in the DataFrame
    available_columns = [
        col for col in desired_column_order if col in results_df.columns
    ]
    # Add remaining columns that are not MAE columns or desired columns
    remaining_columns = [
        col for col in results_df.columns if col not in available_columns
    ]

    final_column_order = available_columns + remaining_columns

    results_df = results_df[final_column_order]

    # Determine output path
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = os.path.join("results")

    os.makedirs(output_dir, exist_ok=True)

    # Extract task name, split, and paraphrase model info from results
    if results:
        # Get info from first result as representative
        first_result = results[0]

        # Get task type from config datasets
        task_types = list(config.get("datasets", {}).keys())
        task_name = task_types[0] if task_types else "unknown"

        split = config.get("split", "unknown")
        paraphrase_model = first_result.get("gen_model", "no-paraphrase")

        # Sanitize paraphrase model name for filename (replace : with -)
        if paraphrase_model and paraphrase_model != "no-paraphrase":
            paraphrase_model = paraphrase_model.replace(":", "-")

        filename = f"{timestamp}_{task_name}_{split}_{paraphrase_model}"
    else:
        filename = f"{timestamp}_ParaMTEB_Eval_Results"

    output_path = os.path.join(output_dir, filename)

    # Save detailed results in JSONL format
    results_df.to_json(f"{output_path}.jsonl", orient="records", lines=True)
    print(f"Detailed results saved to {output_path}.jsonl")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="PTEB Evaluation Framework - Paraphrase-augmented MTEB benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset (task auto-detected, embedding model required)
  python -m pteb.pteb_eval --datasets stsbenchmark --emb_model all-mpnet-base-v2

  # Multiple datasets with 3 paraphrase runs (tasks auto-detected and grouped)
  python -m pteb.pteb_eval --datasets stsbenchmark sts12 sts13 --emb_model all-mpnet-base-v2 --n_runs 3

  # Filter multilingual dataset to English only (monolingual datasets unaffected)
  python -m pteb.pteb_eval --datasets biosses sts17-crosslingual --emb_model all-mpnet-base-v2 --languages EN

  # Multiple languages for crosslingual datasets
  python -m pteb.pteb_eval --datasets sts17-crosslingual --emb_model all-mpnet-base-v2 --languages EN FR DE

  # Override generation parameters (temperature, top_p)
  python -m pteb.pteb_eval --datasets stsbenchmark --emb_model all-mpnet-base-v2 --temperature 0.7 --top_p 0.95

  # Memory-constrained environment (reduce batch sizes)
  python -m pteb.pteb_eval --datasets stsbenchmark --emb_model all-mpnet-base-v2 --emb_batch_size 512 --encoding_chunk_size 512

  # Custom seeds for reproducibility
  python -m pteb.pteb_eval --datasets stsbenchmark --emb_model all-mpnet-base-v2 --seed_paraphrasing 42 --seed_general 42
        """,
    )

    # Required arguments
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="[REQUIRED] Dataset name(s) as space-separated list (e.g., 'stsbenchmark' or 'stsbenchmark sts12 sts13'). Task type is auto-detected from dataset metadata.",
    )
    parser.add_argument(
        "--emb_model",
        type=str,
        required=True,
        help="[REQUIRED] Embedding model name (e.g., 'all-mpnet-base-v2', 'all-MiniLM-L6-v2')",
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of paraphrase runs (default: 1) - controls how many paraphrased versions are generated per sentence",
    )
    parser.add_argument(
        "--gen_model",
        type=str,
        default="gemma3:27b",
        help="Generative model for paraphrasing (default: gemma3:27b)",
    )
    parser.add_argument(
        "--save_paraphrases",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=True,
        help="Whether to save generated paraphrases to disk (default: True)",
    )

    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for paraphrase generation (default: 0.8, range: 0.0-2.0)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter (default: 0.9, range: 0.0-1.0)",
    )
    parser.add_argument(
        "--paraphrase_prompt",
        type=str,
        default=None,
        help="Custom prompt for paraphrase generation (default: None, uses model-specific defaults)",
    )

    # Batch and memory parameters
    parser.add_argument(
        "--emb_batch_size",
        type=int,
        default=4096,
        help="Batch size for embedding computation (default: 4096)",
    )
    parser.add_argument(
        "--encoding_chunk_size",
        type=int,
        default=4096,
        help="Maximum sentences per encoding chunk (default: 4096, smaller = less memory)",
    )
    parser.add_argument(
        "--clustering_batch_size",
        type=int,
        default=500,
        help="Batch size for Mini-Batch K-Means clustering (default: 500)",
    )
    parser.add_argument(
        "--normalize_embeddings",
        action="store_true",
        default=False,
        help="Enable L2 normalization for embeddings (default: False, recommended for embeddinggemma)",
    )

    parser.add_argument(
        "--st_dtype",
        type=str,
        default="torch.bfloat16",
        choices=["torch.float16", "torch.float32", "torch.bfloat16"],
        help="Data type for SentenceTransformers models (default: torch.bfloat16)",
    )

    # Random seeds
    parser.add_argument(
        "--seed_paraphrasing",
        type=int,
        default=None,
        help="Random seed for paraphrase generation (default: random)",
    )
    parser.add_argument(
        "--seed_general",
        type=int,
        default=None,
        help="Random seed for general operations (default: random)",
    )

    # Downsampling
    parser.add_argument(
        "--downsample",
        type=float,
        default=None,
        help="Fraction of data to use (0.0-1.0). For retrieval, downsamples queries only.",
    )

    # Language filtering (for multilingual datasets)
    parser.add_argument(
        "--languages",
        type=str,
        nargs="*",
        default=None,
        help="Language codes (ISO 639-3) to filter multilingual datasets (e.g., 'ENG SPA FRA' or 'EN ES FR'). Only applies to crosslingual datasets like STS17, STS22. Monolingual datasets ignore this parameter. Supports both 2-letter (EN) and 3-letter (ENG) codes. Default: None (uses all available languages)",
    )

    args = parser.parse_args()

    return args


def main(cli_args: argparse.Namespace) -> None:
    """
    Main evaluation function using the class-based framework.

    Args:
        cli_args: Command-line arguments
    """
    print("PTEB Evaluation Framework")
    print("=" * 50)

    # Build configuration dictionary from CLI arguments
    # Auto-detect task types and group datasets
    dataset_names = cli_args.datasets

    # Auto-detect task type for each dataset and group by task
    task_datasets = {}  # {task_type: [dataset1, dataset2, ...]}
    for dataset_name in dataset_names:
        try:
            task_type = get_task_type_from_dataset(dataset_name)
            if task_type not in task_datasets:
                task_datasets[task_type] = []
            task_datasets[task_type].append(dataset_name)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Validate downsample argument
    if cli_args.downsample is not None:
        if cli_args.downsample <= 0.0 or cli_args.downsample > 1.0:
            print("Error: --downsample must be in range (0.0, 1.0]")
            sys.exit(1)

    # Print summary of auto-detection
    print("\nAuto-detected task types and grouped datasets:")
    for task_type, datasets in task_datasets.items():
        print(f"  {task_type}: {datasets}")

    # Build config dictionary from CLI arguments
    config = {
        # Dataset configuration
        "datasets": task_datasets,
        # Model configuration
        "emb_model": cli_args.emb_model,
        "gen_model": cli_args.gen_model,
        # Generation parameters
        "n_runs": cli_args.n_runs,
        "temperature": cli_args.temperature,
        "top_p": cli_args.top_p,
        "paraphrase_prompt": cli_args.paraphrase_prompt,
        "save_paraphrases": cli_args.save_paraphrases,
        # Batch and memory parameters
        "emb_batch_size": cli_args.emb_batch_size,
        "encoding_chunk_size": cli_args.encoding_chunk_size,
        "clustering_batch_size": cli_args.clustering_batch_size,
        "normalize_embeddings": cli_args.normalize_embeddings,
        "st_dtype": cli_args.st_dtype,
        # Seed parameters
        "seed_paraphrasing": cli_args.seed_paraphrasing,
        "seed_general": cli_args.seed_general,
        # Downsampling
        "downsample": cli_args.downsample,
        # Language filtering (for multilingual datasets)
        "languages": cli_args.languages,
        # Default values for parameters not in CLI
        "emb_model_api": "sentence-transformers",
        "split": "test",
    }

    # Normalize language codes to lowercase for consistent matching
    # MTEB uses lowercase codes (e.g., "en", "en-en", "fr", "de")
    if config["languages"] is not None:
        config["languages"] = [lang.lower() for lang in config["languages"]]
        print(f"\nLanguage filter applied: {config['languages']}")

    # Validate configuration
    validate_config(config)

    start_time = time.time()

    # Set torch backends for optimal performance
    if torch.cuda.is_available():
        # Enable TensorFloat-32 for matrix multiplications on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        # Enable cuDNN auto-tuning for optimal algorithm selection
        torch.backends.cudnn.benchmark = True
        print("Enabled torch.backends optimizations (TF32, cuDNN benchmark)")

    # Check embedding model configuration
    emb_model = config.get("emb_model", "")
    if not isinstance(emb_model, str) or not emb_model:
        print("Error: Embedding model must be specified via --emb_model.")
        return

    print(f"Using embedding model: {emb_model}")

    clear_gpu_memory()

    # Load embedding model
    print(f"\n{'=' * 80}")
    print(f"Loading embedding model: {emb_model}")
    print(f"{'=' * 80}")

    embedding_model: SentenceTransformer = create_embedding_model(
        model_name=emb_model, config=config
    )

    # Initialize results collection
    results = []

    # Task mapping for dynamic task instantiation
    task_mapping: dict[str, dict[str, Any]] = {
        "sts": {"class": STSTask, "display_name": "STS"},
        "pair_classification": {
            "class": PairClassificationTask,
            "display_name": "PairClassification",
        },
        "reranking": {"class": RerankingTask, "display_name": "Reranking"},
        "clustering": {"class": ClusteringTask, "display_name": "Clustering"},
        "retrieval": {"class": RetrievalTask, "display_name": "Retrieval"},
        "summarization": {"class": SummarizationTask, "display_name": "Summarization"},
        "classification": {
            "class": ClassificationTask,
            "display_name": "Classification",
        },
    }

    failed_datasets = []

    assert isinstance(config["datasets"], dict)
    # Iterate through configured datasets and run corresponding tasks
    for task_type, datasets in config["datasets"].items():
        if task_type in task_mapping:
            try:
                task_info = task_mapping[task_type]
                task_class = task_info["class"]
                display_name = task_info["display_name"]

                print_task_hyperparameters(display_name, config)
                print(f"Initializing {display_name} task...")

                # Dynamically instantiate the task
                task_instance = task_class(datasets=datasets, config=config)

                print(f"{display_name} Task configured with datasets: {datasets}")

                # Run evaluation
                print(f"\nRunning {display_name} evaluation for {emb_model}...")
                task_results = task_instance.run_full_evaluation(
                    [embedding_model],
                    [emb_model],
                    model_max_seq_length=getattr(
                        embedding_model, "max_seq_length", None
                    ),
                )
                results.extend(task_results)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("Caught CUDA OOM error: ", e)
                    # Optionally free up cached memory
                    torch.cuda.empty_cache()
                failed_datasets.append((task_type, datasets, str(e)))
            clear_gpu_memory()
            print("-" * 80)
            print()

    # Clean up memory
    print(f"\nCompleted evaluation. Results: {len(results)} evaluations")
    del embedding_model
    clear_gpu_memory()

    for result in results:
        # Multiple original/paraphrased main score fields by 100
        if (
            "original_main_score" in result
            and result["original_main_score"] is not None
        ):
            result["original_main_score"] *= 100
        if (
            "paraphrased_main_score" in result
            and result["paraphrased_main_score"] is not None
        ):
            result["paraphrased_main_score"] *= 100

    # Calculate total time
    total_time = time.time() - start_time

    # Print summary
    print_evaluation_summary(results, total_time)

    # Save results (always save in JSONL format)
    print("\nSaving results...")
    save_evaluation_results(results, config)

    print("\nEvaluation completed!")

    if failed_datasets:
        print("\nSome datasets failed during evaluation:")
        for task_type, datasets, error_msg in failed_datasets:
            print(f"  - Task: {task_type}, Datasets: {datasets}, Error: {error_msg}")

    # Play completion sound
    play_completion_sound()


def cli_entry_point() -> None:
    """
    Entry point for console script (pteb command).

    Supports both evaluation mode and discovery subcommands:
        pteb --datasets stsbenchmark --emb_model all-mpnet-base-v2  (evaluation)
        pteb get-tasks                                               (list tasks)
        pteb get-datasets sts                                        (list datasets)
        pteb get-metadata rte3                                       (show metadata)
    """
    import sys
    from pteb.dataset_metadata import get_tasks, get_datasets, get_dataset_metadata

    # Check for subcommands
    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "get-tasks":
            for task in get_tasks():
                print(task)
            return

        if cmd == "get-datasets":
            if len(sys.argv) < 3:
                print("Usage: pteb get-datasets <task_name>")
                print(f"Available tasks: {', '.join(get_tasks())}")
                sys.exit(1)
            task_name = sys.argv[2]
            datasets = get_datasets(task_name)
            if not datasets:
                print(f"No datasets found for task '{task_name}'")
                print(f"Available tasks: {', '.join(get_tasks())}")
                sys.exit(1)
            for ds in datasets:
                print(ds)
            return

        if cmd == "get-metadata":
            if len(sys.argv) < 3:
                print("Usage: pteb get-metadata <dataset_name>")
                sys.exit(1)
            dataset_name = sys.argv[2]
            try:
                meta = get_dataset_metadata(dataset_name)
                print(f"name: {meta.name}")
                print(f"alternate_names: {meta.alternate_names}")
                print(f"hf_dataset_name: {meta.hf_dataset_name}")
                print(f"task_type: {meta.task_type}")
                print(f"metric: {meta.metric}")
                print(f"languages: {meta.languages}")
                print(f"description: {meta.description}")
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)
            return

    # Default: run evaluation
    args = parse_arguments()
    main(cli_args=args)


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Call main with parsed arguments
    main(cli_args=args)
