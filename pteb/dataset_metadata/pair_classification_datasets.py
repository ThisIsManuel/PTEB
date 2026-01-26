"""
Pair Classification dataset metadata classes.

This module contains metadata for all pair classification datasets used in PTEB evaluations.
"""

from .base_dataset_metadata import BaseDatasetMetadata


class TwitterSemEval2015Metadata(BaseDatasetMetadata):
    """Twitter SemEval 2015 pair classification dataset."""

    name = "twittersemeval2015-pairclassification"
    alternate_names: list[str] = ["twittersemeval2015"]
    hf_dataset_name = "mteb/twittersemeval2015-pairclassification"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["eng"]  # Monolingual English
    task_type = "pair_classification"
    metric = "ap"
    description = (
        "Twitter SemEval 2015 dataset - English social/written text (16,777 samples)"
    )
    eval_split = "test"
    train_split = None


class RTE3Metadata(BaseDatasetMetadata):
    """RTE3 (Recognizing Textual Entailment) multilingual dataset."""

    name = "rte3"
    alternate_names: list[str] = []
    hf_dataset_name = "mteb/rte3"
    default_splits: list[str] = ["test"]
    languages = [
        "de",  # German
        "en",  # English
        "fr",  # French
        "it",  # Italian
    ]
    task_type = "pair_classification"
    metric = "ap"
    description = "RTE3 multilingual textual entailment dataset (de, en, fr, it)"
    eval_split = "test"
    train_split = None
