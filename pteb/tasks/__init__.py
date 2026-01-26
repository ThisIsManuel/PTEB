"""
Task implementations for PTEB.

This module contains task classes that define how different evaluation
tasks (STS, classification, clustering, etc.) are executed.
"""

from .base_task import BaseTask
from .classification_task import ClassificationTask
from .clustering_task import ClusteringTask
from .pair_classification_task import PairClassificationTask
from .paraphrase_task import ParaphraseTask
from .reranking_task import RerankingTask
from .retrieval_task import RetrievalTask
from .sts_task import STSTask
from .summarization_task import SummarizationTask

__all__ = [
    "BaseTask",
    "ParaphraseTask",
    "STSTask",
    "PairClassificationTask",
    "ClusteringTask",
    "SummarizationTask",
    "RetrievalTask",
    "RerankingTask",
    "ClassificationTask",
]
