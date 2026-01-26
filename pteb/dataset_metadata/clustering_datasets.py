"""
Clustering dataset metadata classes.

This module contains metadata for all clustering datasets used in PTEB evaluations.
"""

from .base_dataset_metadata import BaseDatasetMetadata


class TwentyNewsgroupsClusteringMetadata(BaseDatasetMetadata):
    """
    20 Newsgroups dataset for clustering evaluation.

    A monolingual English dataset containing newsgroup documents
    categorized into 20 different topics.
    """

    name = "twentynewsgroups-clustering"
    alternate_names: list[str] = [
        "twentynewsgroups",
        "20newsgroups",
        "20newsgroups-clustering",
    ]
    hf_dataset_name = "mteb/twentynewsgroups-clustering"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["eng"]  # Monolingual English
    task_type = "clustering"
    metric = "v_measure"
    description = "20 Newsgroups text classification dataset for clustering evaluation"
    eval_split = "test"
    train_split = None
    clustering_batch_size = 500


class MasakhaNEWSClusteringS2SMetadata(BaseDatasetMetadata):
    """
    MasakhaNEWS multilingual African news clustering dataset.

    Contains news articles in multiple African languages for
    clustering evaluation.
    """

    name = "MasakhaNEWSClusteringS2S"
    alternate_names: list[str] = [
        "masakhanews",
        "masakhanews-clustering",
        "masakhanews-s2s",
    ]
    hf_dataset_name = "mteb/MasakhaNEWSClusteringS2S"
    default_splits: list[str] = ["test"]
    languages = [
        "amh",  # Amharic
        "eng",  # English
        "fra",  # French
        "hau",  # Hausa
        "ibo",  # Igbo
        "lin",  # Lingala
        "lug",  # Luganda
        "orm",  # Oromo
        "pcm",  # Nigerian Pidgin
        "run",  # Kirundi
        "sna",  # Shona
        "som",  # Somali
        "swa",  # Swahili
        "tir",  # Tigrinya
        "xho",  # Xhosa
    ]
    task_type = "clustering"
    metric = "v_measure"
    description = "Multilingual African news articles clustering dataset"
    eval_split = "test"
    train_split = None
    clustering_batch_size = 500


class HALClusteringS2SV2Metadata(BaseDatasetMetadata):
    """
    HAL (Hyper Articles en Ligne) clustering dataset.

    Contains French academic articles for clustering evaluation.
    """

    name = "HALClusteringS2S.v2"
    alternate_names: list[str] = ["halclustering", "halclusterings2s"]
    hf_dataset_name = "mteb/HALClusteringS2S.v2"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["fra"]  # French only
    task_type = "clustering"
    metric = "v_measure"
    description = "French academic articles clustering dataset from HAL"
    eval_split = "test"
    train_split = None
    clustering_batch_size = 500
