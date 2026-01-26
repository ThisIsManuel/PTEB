"""
Abstract base class for dataset metadata in PTEB.

This module defines the BaseDatasetMetadata class that encapsulates
dataset-specific properties like HuggingFace dataset paths, available splits,
supported languages, and task types. This follows MTEB's TaskMetadata pattern.
"""

from abc import ABC
from typing import Any


class BaseDatasetMetadata(ABC):
    """
    Abstract base class for dataset metadata.

    Each concrete dataset metadata class should define these class attributes:
        - name: Canonical dataset name (e.g., "twentynewsgroups-clustering")
        - hf_dataset_name: HuggingFace Hub dataset path (e.g., "mteb/twentynewsgroups-clustering")
        - default_splits: List of available splits (e.g., ["test"], ["train", "validation", "test"])
        - languages: 3-letter ISO 639-2/3 language codes (e.g., ["eng"], ["fra"], ["deu"])
                    Use language pairs for cross-lingual tasks (e.g., ["eng-fra"], ["deu-eng"])
                    None treated as unknown language (conservative: triggers non-English prompts)
        - task_type: Task category (e.g., "clustering", "sts", "pair_classification")
        - description: Human-readable description of the dataset
        - loading_config: Optional dict with dataset-specific loading configuration
        - score_range: Optional tuple (min, max) for STS datasets to normalize similarity scores
        - eval_split: Split to use for evaluation (e.g., "test", "train")
        - train_split: Optional split to use for training (only for classification tasks)

    Use is_non_en() classmethod to determine if dataset contains non-English data.

    Loading Config Options:
        None: Simple loading (most STS, Clustering, Pair Classification datasets)
            load_dataset("mteb/dataset-name", split="test")

        {"requires_language_config": True, "default_language": "en"}:
            Classification datasets that require language specification
            load_dataset("mteb/dataset-name", "en", split="test")

        {"config_names": ["corpus", "queries", "default"]}:
            Retrieval/Reranking datasets requiring multiple named configs
            load_dataset("mteb/dataset-name", name="corpus", split="test")

        {"config_names": [...], "corpus_split": "corpus", "queries_split": "queries", "qrels_split": "train"}:
            Retrieval datasets with non-standard split names
            - corpus_split: Split name for corpus data (default: "corpus")
            - queries_split: Split name for queries data (default: "queries")
            - qrels_split: Split name for relevance judgments (default: eval_split or "test")
            - default_split: Fallback split name for qrels if qrels_split not specified

    Example:
        class TwentyNewsgroupsClusteringMetadata(BaseDatasetMetadata):
            name = "twentynewsgroups-clustering"
            hf_dataset_name = "mteb/twentynewsgroups-clustering"
            default_splits = ["test"]
            languages = ["eng"]  # Monolingual English
            task_type = "clustering"
            description = "20 Newsgroups dataset for clustering evaluation"
            loading_config = None  # Simple loading
            score_range = None  # Not an STS dataset
            # is_non_en() returns False (English-only)

        class STSBenchmarkMetadata(BaseDatasetMetadata):
            name = "stsbenchmark-sts"
            hf_dataset_name = "mteb/stsbenchmark-sts"
            default_splits = ["test", "dev", "train"]
            languages = ["eng"]  # Monolingual English
            task_type = "sts"
            description = "STS Benchmark dataset"
            loading_config = None
            score_range = (0, 5)  # STS scores from 0 to 5
            eval_split = "test"  # Split to use for evaluation
            train_split = None  # Not a classification task
            # is_non_en() returns False (English-only)

        class MasakhaNEWSClusteringS2SMetadata(BaseDatasetMetadata):
            name = "MasakhaNEWSClusteringS2S"
            hf_dataset_name = "mteb/MasakhaNEWSClusteringS2S"
            default_splits = ["test"]
            languages = ["amh", "eng", "fra", ...]  # Multiple African languages
            task_type = "clustering"
            description = "Multilingual African news clustering"
            # is_non_en() returns True (multilingual)
    """

    # Required class attributes (must be defined in subclasses)
    name: str
    hf_dataset_name: str
    default_splits: list[str]
    languages: list[str] | None
    task_type: str
    metric: str
    description: str
    alternate_names: list[
        str
    ] = []  # Alternative names for matching (e.g., ["stsb", "stsbenchmark"])
    loading_config: dict[str, Any] | None = (
        None  # Dataset-specific loading configuration
    )
    score_range: tuple[float, float] | None = (
        None  # (min, max) for STS score normalization
    )
    eval_split: str = "test"  # Split to use for evaluation
    train_split: str | None = (
        None  # Split to use for training (classification tasks only)
    )

    @classmethod
    def is_non_en(cls) -> bool:
        """
        Determine if dataset contains non-English data based on languages list.

        Uses a conservative approach: datasets are treated as non-English unless
        explicitly marked with English-only language codes.

        Returns True (non-English) for:
        - languages = None (unknown language - conservative approach)
        - languages = [] (empty list - conservative approach)
        - languages = ["fra"], ["deu"], ["ara"] (non-English languages)
        - languages = ["eng", "fra"] (multilingual)
        - languages = ["eng-fra"] (cross-lingual pair)

        Returns False (English-only) for:
        - languages = ["eng"], ["eng-eng"], ["eng-ext"] (3-letter codes)
        - Legacy: ["en"], ["en-en"] (2-letter codes, backward compatibility)

        Returns:
            True if dataset contains non-English data or language is unknown,
            False only if explicitly English-only

        Examples:
            >>> TwentyNewsgroupsClusteringMetadata.is_non_en()
            False  # languages = ["eng"]
            >>> MasakhaNEWSClusteringS2SMetadata.is_non_en()
            True  # languages = ["amh", "eng", "fra", ...]
            >>> HALClusteringS2SV2Metadata.is_non_en()
            True  # languages = ["fra"]
        """
        # None or empty list → treat as non-English (conservative approach)
        if cls.languages is None or cls.languages == []:
            return True

        # Check if ALL languages are English variants (both 2-letter and 3-letter codes)
        # Note: We support both for backward compatibility, but 3-letter is preferred
        english_variants = {"en", "en-en", "eng", "eng-eng", "eng-ext"}
        languages_set = {lang.lower() for lang in cls.languages}

        # If all languages are English variants → English-only
        if languages_set.issubset(english_variants):
            return False

        # Otherwise → contains non-English languages
        return True

    @classmethod
    def is_monolingual(cls) -> bool:
        """
        Determine if dataset is truly monolingual (single language only).

        For clustering tasks: If True, each row in the dataset should be treated
        as a separate clustering problem WITHOUT regrouping by language.

        Returns True (monolingual) only if:
        - languages = ["eng"], ["fra"], ["deu"], etc. (single language)
        - languages = ["eng-eng"], ["ara-ara"], ["rus-rus"] (same language pairs)

        Returns False (multilingual) for:
        - languages = None (unknown - conservative)
        - languages = [] (empty - conservative)
        - languages = ["eng", "fra"] (multiple different languages)
        - languages = ["eng-fra"], ["deu-eng"] (cross-lingual pairs)
        - Any list with more than one distinct language

        Returns:
            True if dataset contains exactly one language, False otherwise

        Examples:
            >>> TwentyNewsgroupsClusteringMetadata.is_monolingual()
            True  # languages = ["eng"]
            >>> HALClusteringS2SV2Metadata.is_monolingual()
            True  # languages = ["fra"]
            >>> BIOSSESMetadata.is_monolingual()
            True  # languages = ["eng"] (monolingual English)
            >>> MasakhaNEWSClusteringS2SMetadata.is_monolingual()
            False  # languages = ["amh", "eng", "fra", ...]
        """
        # None or empty list → treat as NOT monolingual (conservative)
        if cls.languages is None or cls.languages == []:
            return False

        # Single language code → monolingual
        if len(cls.languages) == 1:
            lang = cls.languages[0].lower()

            # Check if it's a language pair (e.g., "eng-eng", "ar-ar")
            if "-" in lang:
                parts = lang.split("-")
                # If both parts are the same, it's still monolingual
                # e.g., "eng-eng" → monolingual, "en-fr" → NOT monolingual
                if len(parts) == 2 and parts[0] == parts[1]:
                    return True
                else:
                    # Cross-lingual pair like "en-fr" → NOT monolingual
                    return False

            # Simple language code → monolingual
            return True

        # Multiple languages → NOT monolingual
        return False

    @classmethod
    def matches(cls, dataset_name: str) -> bool:
        """
        Check if the given dataset_name matches this metadata class.

        The matching is case-insensitive and removes common separators
        (dashes, underscores) to handle various naming conventions.
        Also checks alternate_names for additional matching options.

        Args:
            dataset_name: Name of the dataset to match (e.g., "TwentyNewsgroupsClustering",
                         "twentynewsgroups-clustering", "twentynewsgroups")

        Returns:
            True if dataset_name matches this metadata class, False otherwise

        Examples:
            >>> TwentyNewsgroupsClusteringMetadata.matches("twentynewsgroups-clustering")
            True
            >>> TwentyNewsgroupsClusteringMetadata.matches("TwentyNewsgroupsClustering")
            True
            >>> STSBenchmarkMetadata.matches("stsb")
            True  # matches via alternate_names
            >>> TwentyNewsgroupsClusteringMetadata.matches("stsbenchmark")
            False
        """
        # Normalize input: lowercase, remove dashes and underscores
        normalized_input = dataset_name.lower().replace("-", "").replace("_", "")

        # Check primary name
        normalized_class = cls.name.lower().replace("-", "").replace("_", "")
        if normalized_input == normalized_class:
            return True

        # Check alternate names
        for alt_name in cls.alternate_names:
            normalized_alt = alt_name.lower().replace("-", "").replace("_", "")
            if normalized_input == normalized_alt:
                return True

        return False

    def get_split(self, requested_split: str) -> str:
        """
        Validate that the requested split exists and return it.

        Args:
            requested_split: Name of the split to validate (e.g., "test", "train")

        Returns:
            The validated split name

        Raises:
            ValueError: If the requested split is not available for this dataset

        Examples:
            >>> metadata = TwentyNewsgroupsClusteringMetadata()
            >>> metadata.get_split("test")
            'test'
            >>> metadata.get_split("train")
            ValueError: Split 'train' not available for twentynewsgroups-clustering
        """
        if requested_split not in self.default_splits:
            raise ValueError(
                f"Split '{requested_split}' not available for {self.name}. "
                f"Available splits: {self.default_splits}"
            )
        return requested_split

    def get_languages(self) -> list[str] | None:
        """
        Get the list of supported languages for this dataset.

        Returns:
            List of language codes if multilingual, None if monolingual English

        Examples:
            >>> metadata = TwentyNewsgroupsClusteringMetadata()
            >>> metadata.get_languages()
            None  # Monolingual English

            >>> metadata = MasakhaNEWSClusteringMetadata()
            >>> metadata.get_languages()
            ['amh', 'eng', 'fra', ...]  # Multilingual
        """
        return self.languages

    def __repr__(self) -> str:
        """String representation of the metadata."""
        lang_str = (
            f"{len(self.languages)} languages" if self.languages else "monolingual"
        )
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"task_type='{self.task_type}', "
            f"splits={self.default_splits}, "
            f"{lang_str})"
        )
