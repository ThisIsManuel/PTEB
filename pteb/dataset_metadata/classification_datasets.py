"""
Classification dataset metadata classes.

This module contains metadata for all classification datasets used in PTEB evaluations.
"""

from .base_dataset_metadata import BaseDatasetMetadata


class Banking77Metadata(BaseDatasetMetadata):
    """Banking77 intent classification dataset."""

    name = "banking77"
    alternate_names: list[str] = []
    hf_dataset_name = "mteb/banking77"
    default_splits: list[str] = ["test", "train"]
    languages: list[str] = ["eng"]  # Monolingual English
    task_type = "classification"
    metric = "accuracy"
    description = "Banking77 intent classification dataset - English banking queries"
    eval_split = "test"
    train_split = "train"  # Classification task needs training data
    loading_config = None  # Simple loading without config


class AmazonCounterfactualMetadata(BaseDatasetMetadata):
    """Amazon Counterfactual classification dataset - multilingual."""

    name = "amazon_counterfactual"
    alternate_names: list[str] = []
    hf_dataset_name = "mteb/amazon_counterfactual"
    default_splits: list[str] = ["test", "train", "validation"]
    languages = [
        "eng",  # English
        "eng-ext",  # English Extended
        "deu",  # German
        "jpn",  # Japanese
    ]
    task_type = "classification"
    metric = "accuracy"
    description = "Amazon Counterfactual classification dataset - multilingual (EN, EN-EXT, DE, JA)"
    eval_split = "test"
    train_split = "train"  # Classification task needs training data
    loading_config = {"requires_language_config": True, "default_language": "eng"}
