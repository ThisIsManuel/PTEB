"""
Retrieval dataset metadata classes.

This module contains metadata for all retrieval datasets used in PTEB evaluations.
"""

from .base_dataset_metadata import BaseDatasetMetadata


class ArguAnaMetadata(BaseDatasetMetadata):
    """ArguAna argument retrieval dataset."""

    name = "arguana"
    alternate_names: list[str] = []
    hf_dataset_name = "mteb/arguana"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["eng"]  # Monolingual English
    task_type = "retrieval"
    metric = "ndcg@10"
    description = "ArguAna argument retrieval dataset - English medical/written"
    eval_split = "test"
    train_split = None
    loading_config = {
        "config_names": ["corpus", "queries", "default"],
        "corpus_split": "corpus",  # Use "corpus" split instead of "test"
        "queries_split": "queries",  # Use "queries" split instead of "test"
    }


class TwitterHjerneRetrievalMetadata(BaseDatasetMetadata):
    """TwitterHjerne Danish Twitter retrieval dataset."""

    name = "TwitterHjerneRetrieval"
    alternate_names: list[str] = ["twitterhjerne"]
    hf_dataset_name = "mteb/TwitterHjerneRetrieval"
    default_splits: list[str] = ["train"]  # Only train split available
    languages: list[str] = ["dan"]  # Danish
    task_type = "retrieval"
    metric = "ndcg@10"
    description = "TwitterHjerne Danish Twitter retrieval dataset"
    eval_split = "train"  # Special case: uses train split for evaluation
    train_split = None
    loading_config = {
        "config_names": ["corpus", "queries", "default"],
        "corpus_split": "train",  # Special case: all configs use train split
        "queries_split": "train",
        "qrels_split": "train",
    }


class LegalQuADMetadata(BaseDatasetMetadata):
    """LegalQuAD German legal question answering dataset."""

    name = "LegalQuAD"
    alternate_names: list[str] = []
    hf_dataset_name = "mteb/LegalQuAD"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["deu"]  # German
    task_type = "retrieval"
    metric = "ndcg@10"
    description = "LegalQuAD German legal question answering retrieval dataset"
    eval_split = "test"
    train_split = None
    loading_config = {
        "config_names": ["corpus", "queries", "default"],
        "corpus_split": "corpus",
        "queries_split": "queries",
    }
