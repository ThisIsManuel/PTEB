"""
This module contains metadata for all datasets used in PTEB evaluations.
"""

from .base_dataset_metadata import BaseDatasetMetadata


class STSBenchmarkMetadata(BaseDatasetMetadata):
    """STS Benchmark dataset - standard English STS evaluation."""

    name = "stsbenchmark-sts"
    alternate_names: list[str] = ["stsbenchmark", "stsb-sts", "stsb"]
    hf_dataset_name = "mteb/stsbenchmark-sts"
    default_splits: list[str] = ["test", "dev", "train"]
    languages: list[str] = ["eng"]
    task_type = "sts"
    metric = "spearman"
    description = "STS Benchmark dataset - English blog/news/written text"
    eval_split = "test"
    train_split = None
    score_range = (0, 5)


class STS12Metadata(BaseDatasetMetadata):
    """STS 2012 dataset - English encyclopedic and news text."""

    name = "sts12-sts"
    alternate_names: list[str] = ["sts12"]
    hf_dataset_name = "mteb/sts12-sts"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["eng"]
    task_type = "sts"
    metric = "spearman"
    description = "STS 2012 dataset - English encyclopedic/news/written text"
    eval_split = "test"
    train_split = None
    score_range = (0, 5)


class STS13Metadata(BaseDatasetMetadata):
    """STS 2013 dataset - English web and news text."""

    name = "sts13-sts"
    alternate_names: list[str] = ["sts13"]
    hf_dataset_name = "mteb/sts13-sts"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["eng"]
    task_type = "sts"
    metric = "spearman"
    description = "STS 2013 dataset - English web/news/non-fiction text"
    eval_split = "test"
    train_split = None
    score_range = (0, 5)


class STS14Metadata(BaseDatasetMetadata):
    """STS 2014 dataset - English blog, web, and spoken text."""

    name = "sts14-sts"
    alternate_names: list[str] = ["sts14"]
    hf_dataset_name = "mteb/sts14-sts"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["eng"]
    task_type = "sts"
    metric = "spearman"
    description = "STS 2014 dataset - English blog/web/spoken text"
    eval_split = "test"
    train_split = None
    score_range = (0, 5)


class STS15Metadata(BaseDatasetMetadata):
    """STS 2015 dataset - English blog, news, and web text."""

    name = "sts15-sts"
    alternate_names: list[str] = ["sts15"]
    hf_dataset_name = "mteb/sts15-sts"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["eng"]
    task_type = "sts"
    metric = "spearman"
    description = "STS 2015 dataset - English blog/news/web text"
    eval_split = "test"
    train_split = None
    score_range = (0, 5)


class STS16Metadata(BaseDatasetMetadata):
    """STS 2016 dataset - English text from multiple domains."""

    name = "sts16-sts"
    alternate_names: list[str] = ["sts16"]
    hf_dataset_name = "mteb/sts16-sts"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["eng"]
    task_type = "sts"
    metric = "spearman"
    description = "STS 2016 dataset - English text from multiple domains"
    eval_split = "test"
    train_split = None
    score_range = (0, 5)


class STS17Metadata(BaseDatasetMetadata):
    """STS 2017 crosslingual dataset - 11 language pairs."""

    name = "sts17-crosslingual-sts"
    alternate_names: list[str] = ["sts17", "sts17-crosslingual"]
    hf_dataset_name = "mteb/sts17-crosslingual-sts"
    default_splits: list[str] = ["test"]
    languages = [
        "ara-ara",  # Arabic-Arabic
        "ara-eng",  # Arabic-English
        "eng-ara",  # English-Arabic
        "eng-deu",  # English-German
        "eng-eng",  # English-English
        "eng-tur",  # English-Turkish
        "spa-eng",  # Spanish-English
        "spa-spa",  # Spanish-Spanish
        "fra-eng",  # French-English
        "ita-eng",  # Italian-English
        "nld-eng",  # Dutch-English
    ]
    task_type = "sts"
    metric = "spearman"
    description = "STS 2017 crosslingual dataset - 11 language pairs, news/web/written"
    eval_split = "test"
    train_split = None
    score_range = (0, 5)


class STS22Metadata(BaseDatasetMetadata):
    """STS 2022 crosslingual dataset - 18 language pairs."""

    name = "sts22-crosslingual-sts"
    alternate_names: list[str] = ["sts22", "sts22-crosslingual"]
    hf_dataset_name = "mteb/sts22-crosslingual-sts"
    default_splits: list[str] = ["test"]
    languages = [
        "ara",  # Arabic
        "deu",  # German
        "deu-eng",  # German-English
        "deu-fra",  # German-French
        "deu-pol",  # German-Polish
        "eng",  # English
        "spa",  # Spanish
        "spa-eng",  # Spanish-English
        "spa-ita",  # Spanish-Italian
        "fra",  # French
        "fra-pol",  # French-Polish
        "ita",  # Italian
        "pol",  # Polish
        "pol-eng",  # Polish-English
        "rus",  # Russian
        "tur",  # Turkish
        "zho",  # Chinese
        "zho-eng",  # Chinese-English
    ]
    task_type = "sts"
    metric = "spearman"
    description = "STS 2022 crosslingual dataset - 18 language pairs, news/written"
    eval_split = "test"
    train_split = None
    score_range = (1, 4)  # Different from other STS datasets


class SICKRMetadata(BaseDatasetMetadata):
    """SICK-R dataset - Sentences Involving Compositional Knowledge."""

    name = "sickr-sts"
    alternate_names: list[str] = ["sickr"]
    hf_dataset_name = "mteb/sickr-sts"
    default_splits: list[str] = ["test", "train"]
    languages: list[str] = ["eng"]
    task_type = "sts"
    metric = "spearman"
    description = "SICK-R dataset - English web/written text"
    eval_split = "test"
    train_split = None
    score_range = (1, 5)  # Starts at 1, not 0


class BIOSSESMetadata(BaseDatasetMetadata):
    """BIOSSES dataset - Biomedical Semantic Similarity Estimation."""

    name = "biosses-sts"
    alternate_names: list[str] = ["biosses"]
    hf_dataset_name = "mteb/biosses-sts"
    default_splits: list[str] = ["test"]
    languages: list[str] = ["eng"]
    task_type = "sts"
    metric = "spearman"
    description = "BIOSSES dataset - English biomedical/scientific text"
    eval_split = "test"
    train_split = None
    score_range = (0, 4)  # Biomedical dataset with 0-4 scale
