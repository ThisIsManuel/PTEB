"""
Dataset metadata module for PTEB.

This module provides dataset-specific metadata (HuggingFace paths, splits, languages)
that is separate from runtime configuration. It follows MTEB's pattern of using
TaskMetadata classes to encapsulate dataset properties.

Usage:
    from pteb.dataset_metadata import get_dataset_metadata

    metadata = get_dataset_metadata("twentynewsgroups-clustering")
    print(metadata.hf_dataset_name)  # "mteb/twentynewsgroups-clustering"
    print(metadata.languages)         # None (monolingual English)
"""

from .base_dataset_metadata import BaseDatasetMetadata
from .classification_datasets import (
    AmazonCounterfactualMetadata,
    Banking77Metadata,
)

# Import all dataset metadata classes from task-specific modules
from .clustering_datasets import (
    HALClusteringS2SV2Metadata,
    MasakhaNEWSClusteringS2SMetadata,
    TwentyNewsgroupsClusteringMetadata,
)
from .pair_classification_datasets import (
    RTE3Metadata,
    TwitterSemEval2015Metadata,
)
from .reranking_datasets import (
    AskUbuntuDupQuestionsMetadata,
    RuBQRerankingMetadata,
)
from .retrieval_datasets import (
    ArguAnaMetadata,
    LegalQuADMetadata,
    TwitterHjerneRetrievalMetadata,
)
from .sts_datasets import (
    BIOSSESMetadata,
    SICKRMetadata,
    STS12Metadata,
    STS13Metadata,
    STS14Metadata,
    STS15Metadata,
    STS16Metadata,
    STS17Metadata,
    STS22Metadata,
    STSBenchmarkMetadata,
)
from .summarization_datasets import (
    SummEvalMetadata,
)

# Create registry of all dataset metadata classes
ALL_DATASET_METADATA_CLASSES = [
    # Clustering
    TwentyNewsgroupsClusteringMetadata,
    MasakhaNEWSClusteringS2SMetadata,
    HALClusteringS2SV2Metadata,
    # STS
    STSBenchmarkMetadata,
    STS12Metadata,
    STS13Metadata,
    STS14Metadata,
    STS15Metadata,
    STS16Metadata,
    STS17Metadata,
    STS22Metadata,
    SICKRMetadata,
    BIOSSESMetadata,
    # Pair Classification
    TwitterSemEval2015Metadata,
    RTE3Metadata,
    # Classification
    Banking77Metadata,
    AmazonCounterfactualMetadata,
    # Retrieval
    ArguAnaMetadata,
    TwitterHjerneRetrievalMetadata,
    LegalQuADMetadata,
    # Reranking
    AskUbuntuDupQuestionsMetadata,
    RuBQRerankingMetadata,
    # Summarization
    SummEvalMetadata,
]


def _validate_no_name_overlaps() -> None:
    """
    Validate that no dataset names or alternate names overlap across metadata classes.

    Raises:
        ValueError: If any names overlap between different metadata classes
    """
    # Map normalized names to their owning class
    name_to_class: dict[str, str] = {}

    for metadata_class in ALL_DATASET_METADATA_CLASSES:
        class_name = metadata_class.__name__

        # Collect all names for this class (primary + alternates)
        all_names = [metadata_class.name] + list(metadata_class.alternate_names)

        for name in all_names:
            # Normalize: lowercase, remove dashes and underscores
            normalized = name.lower().replace("-", "").replace("_", "")

            if normalized in name_to_class:
                existing_class = name_to_class[normalized]
                raise ValueError(
                    f"Duplicate dataset name '{name}' (normalized: '{normalized}') "
                    f"found in both {existing_class} and {class_name}"
                )

            name_to_class[normalized] = class_name


# Run validation on module import
_validate_no_name_overlaps()


def get_dataset_metadata(dataset_name: str) -> BaseDatasetMetadata:
    """
    Find and instantiate metadata for a dataset (case-insensitive lookup).

    This function searches through all registered dataset metadata classes
    and returns an instance of the matching class. The matching is case-insensitive
    and handles common name variations (dashes, underscores, etc.).

    Args:
        dataset_name: Name of the dataset (e.g., "twentynewsgroups-clustering",
                     "TwentyNewsgroupsClustering", "twentynewsgroups")

    Returns:
        Instance of the matching dataset metadata class

    Raises:
        ValueError: If no matching dataset metadata is found

    Examples:
        >>> metadata = get_dataset_metadata("twentynewsgroups-clustering")
        >>> metadata.hf_dataset_name
        'mteb/twentynewsgroups-clustering'
        >>> metadata.is_non_en()
        False

        >>> metadata = get_dataset_metadata("MasakhaNEWSClusteringS2S")
        >>> metadata.is_non_en()
        True
        >>> metadata.languages
        ['amh', 'eng', 'fra', ...]
    """
    for metadata_class in ALL_DATASET_METADATA_CLASSES:
        if metadata_class.matches(dataset_name):
            return metadata_class()

    raise ValueError(
        f"Unknown dataset: {dataset_name}. "
        f"Please add a metadata class for this dataset in pteb/dataset_metadata/"
    )


def get_tasks() -> list[str]:
    """List all available task types."""
    return sorted({m.task_type for m in ALL_DATASET_METADATA_CLASSES})


def get_datasets(task_name: str) -> list[str]:
    """List all datasets for a given task type (case-insensitive)."""
    normalized_task = task_name.lower()
    return sorted([
        m.name for m in ALL_DATASET_METADATA_CLASSES
        if m.task_type == normalized_task
    ])


__all__ = [
    "BaseDatasetMetadata",
    "get_dataset_metadata",
    "get_tasks",
    "get_datasets",
    "ALL_DATASET_METADATA_CLASSES",
]
