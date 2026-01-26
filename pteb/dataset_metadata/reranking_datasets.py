"""
Reranking dataset metadata classes.

This module contains metadata for all reranking datasets used in PTEB evaluations.
"""

from .base_dataset_metadata import BaseDatasetMetadata


class AskUbuntuDupQuestionsMetadata(BaseDatasetMetadata):
    """AskUbuntu duplicate questions reranking dataset."""

    name = "askubuntudupquestions-reranking"
    alternate_names: list[str] = ["askubuntu", "askubuntudupquestions"]
    hf_dataset_name = "mteb/askubuntudupquestions-reranking"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["eng"]  # Monolingual English
    task_type = "reranking"
    metric = "map"
    description = (
        "AskUbuntu duplicate questions reranking dataset - English programming/web"
    )
    eval_split = "test"
    train_split = None
    loading_config = {"config_names": ["corpus", "queries", "default", "top_ranked"]}


class RuBQRerankingMetadata(BaseDatasetMetadata):
    """RuBQ Russian question answering reranking dataset."""

    name = "RuBQReranking"
    alternate_names: list[str] = ["rubq"]
    hf_dataset_name = "mteb/RuBQReranking"
    default_splits: list[str] = ["test"]
    languages: list[str] = [
        "rus",  # Russian
        "rus-rus",  # Russian-Russian
    ]
    task_type = "reranking"
    metric = "map"
    description = "RuBQ Russian question answering reranking dataset"
    eval_split = "test"
    train_split = None
    loading_config = {"config_names": ["corpus", "queries", "default", "top_ranked"]}
