"""
Summarization dataset metadata classes.

This module contains metadata for all summarization datasets used in PTEB evaluations.
"""

from .base_dataset_metadata import BaseDatasetMetadata


class SummEvalMetadata(BaseDatasetMetadata):
    """SummEval summarization evaluation dataset."""

    name = "summeval"
    alternate_names: list[str] = []
    hf_dataset_name = "mteb/summeval"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["eng"]  # Monolingual English
    task_type = "summarization"
    metric = "spearman"
    description = (
        "SummEval summarization evaluation dataset - English news/written (100 samples)"
    )
    eval_split = "test"
    train_split = None
