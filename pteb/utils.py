import gc
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def load_all_files(directory: str | Path, file_type: str) -> pd.DataFrame:
    """
    Load and concatenate all files of a specific type from a directory (not subdirectories).

    Args:
        directory: Path to the directory containing files
        file_type: File extension to load (e.g., "csv", "xlsx", "json", "jsonl")

    Returns:
        Concatenated DataFrame with all data from matching files

    Raises:
        FileNotFoundError: If directory doesn't exist or no files of the specified type found
        ValueError: If files cannot be loaded or concatenated
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    file_type = file_type.lower().lstrip(".")
    pattern = f"*.{file_type}"
    matching_files = list(directory_path.glob(pattern))
    matching_files = [f for f in matching_files if not f.name.startswith("~$")]

    if not matching_files:
        raise FileNotFoundError(f"No {file_type} files found in {directory_path}")

    all_dataframes = []
    loaded_files = []

    for file_path in matching_files:
        try:
            if file_type in ["csv"]:
                df = pd.read_csv(file_path)
            elif file_type in ["xlsx", "xls"]:
                df = pd.read_excel(file_path)
            elif file_type in ["json"]:
                df = pd.read_json(file_path)
            elif file_type in ["jsonl"]:
                df = pd.read_json(file_path, lines=True)
            elif file_type in ["tsv"]:
                df = pd.read_csv(file_path, sep="\t")
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            df["source_file"] = file_path.name

            all_dataframes.append(df)
            loaded_files.append(file_path.name)

        except Exception as e:
            print(f"Warning: Could not load {file_path.name}: {e}")
            continue

    if not all_dataframes:
        raise ValueError(f"No valid {file_type} files could be loaded from {directory_path}")

    try:
        concatenated_df = pd.concat(all_dataframes, ignore_index=True)
    except Exception as e:
        raise ValueError(f"Could not concatenate DataFrames: {e}")

    if "seed_paraphrasing" in concatenated_df.columns:
        seed_source_counts = concatenated_df.groupby("seed_paraphrasing")["source_file"].nunique()

        if any(seed_source_counts > 1):
            print("Deduplicating seeds from multiple source files...")

            unique_seeds = concatenated_df["seed_paraphrasing"].unique()
            deduplicated_dfs = []

            for seed in unique_seeds:
                seed_df = concatenated_df[concatenated_df["seed_paraphrasing"] == seed]
                unique_sources = seed_df["source_file"].unique()

                if len(unique_sources) > 1:
                    latest_source = max(unique_sources)
                    print(
                        f"  Seed {seed}: keeping {latest_source}, dropping {[s for s in unique_sources if s != latest_source]}"
                    )
                    deduplicated_dfs.append(seed_df[seed_df["source_file"] == latest_source])
                else:
                    deduplicated_dfs.append(seed_df)

            concatenated_df = pd.concat(deduplicated_dfs, ignore_index=True)
            print(f"Deduplicated to {len(concatenated_df)} rows")

    print(f"Successfully loaded {len(loaded_files)} {file_type} files from {directory_path}")
    print(f"Total records: {len(concatenated_df):,}")
    print(f"Files loaded: {loaded_files}")

    return concatenated_df


def normalize_punctuation_spacing(text: str) -> str:
    """
    Remove extra whitespace before punctuation marks and strip leading/trailing whitespace.

    This function cleans text by removing extraneous spaces that appear before
    various punctuation marks. These spacing issues commonly occur in LLM-generated
    paraphrases and need to be normalized for consistent evaluation.

    Args:
        text: Input text that may contain punctuation spacing issues

    Returns:
        Text with normalized punctuation spacing and stripped leading/trailing whitespace

    Examples:
        >>> normalize_punctuation_spacing("french court : twitter")
        "french court: twitter"

        >>> normalize_punctuation_spacing("( grammar ) the word in a constituent")
        "(grammar) the word in a constituent"

        >>> normalize_punctuation_spacing("cracklin ' rosie by neil diamond .")
        "cracklin' rosie by neil diamond."

        >>> normalize_punctuation_spacing("Hello  :  world  !  This  ( test )  works .")
        "Hello: world! This (test) works."

    Notes:
        - Handles colons, semicolons, commas, periods, exclamation marks, question marks
        - Handles parentheses, brackets, braces
        - Handles apostrophes, quotation marks
        - Removes spaces before closing punctuation
        - Removes spaces after opening punctuation
        - Collapses multiple spaces to single space
        - Removes leading/trailing whitespace and newlines
    """
    if not text or not isinstance(text, str):
        return text

    # Remove spaces before closing punctuation (most common issues)
    text = re.sub(r"\s+:", ":", text)  # " : " → ":"
    text = re.sub(r"\s+;", ";", text)  # " ; " → ";"
    text = re.sub(r"\s+,", ",", text)  # " , " → ","
    text = re.sub(r"\s+\.", ".", text)  # " . " → "."
    text = re.sub(r"\s+!", "!", text)  # " ! " → "!"
    text = re.sub(r"\s+\?", "?", text)  # " ? " → "?"

    # Remove spaces before closing brackets/parentheses/braces
    text = re.sub(r"\s+\)", ")", text)  # " ) " → ")"
    text = re.sub(r"\s+\]", "]", text)  # " ] " → "]"
    text = re.sub(r"\s+\}", "}", text)  # " } " → "}"

    # Remove spaces after opening brackets/parentheses/braces
    text = re.sub(r"\(\s+", "(", text)  # "( " → "("
    text = re.sub(r"\[\s+", "[", text)  # "[ " → "["
    text = re.sub(r"\{\s+", "{", text)  # "{ " → "{"

    # Remove spaces around apostrophes in common contractions
    # Handle patterns like "it 's" → "it's", "don 't" → "don't"
    # But preserve spaces in phrases like "cracklin ' rosie" where ' is not part of a contraction
    text = re.sub(r"(\w)\s+'(\w)", r"\1'\2", text)  # "don 't" → "don't"
    text = re.sub(r"(\w)'\s+(\w)", r"\1'\2", text)  # "don' t" → "don't"

    # Remove spaces before quotes (less common but possible)
    text = re.sub(r"\s+\"", '"', text)  # ' " ' → '"'
    text = re.sub(r"\"\s+", '"', text)  # '" ' → '"'

    # Collapse multiple spaces to single space
    text = re.sub(r" {2,}", " ", text)

    # Strip leading/trailing whitespace and newlines
    return text.strip()


def clean_dataset_name(dataset_name: str) -> str:
    """
    Clean dataset name by removing 'mteb/' prefix and HuggingFace task suffixes.

    This function dynamically detects and removes task-specific suffixes from
    HuggingFace MTEB dataset names, making them suitable for display and file paths.

    Args:
        dataset_name: Raw dataset name from HuggingFace or config
                     (e.g., "mteb/sts12-sts", "biosses-sts")

    Returns:
        Cleaned dataset name suitable for display and file paths
        (e.g., "sts12", "biosses")

    Examples:
        >>> clean_dataset_name("mteb/sts12-sts")
        "sts12"
        >>> clean_dataset_name("mteb/sickr-sts")
        "sick-r"
        >>> clean_dataset_name("mteb/sts17-crosslingual-sts")
        "sts17"
        >>> clean_dataset_name("mteb/twittersemeval2015-pairclassification")
        "twittersemeval2015"
        >>> clean_dataset_name("biosses-sts")
        "biosses"
        >>> clean_dataset_name("mteb/banking77")
        "banking77"

    Notes:
        - Removes "mteb/" prefix if present
        - Removes common MTEB task suffixes (dynamically detected)
        - Special case: converts "sickr" to "sick-r"
        - Works with or without "mteb/" prefix
    """
    # Step 1: Remove "mteb/" prefix if present
    cleaned = dataset_name.replace("mteb/", "")

    # Step 2: Remove common MTEB task suffixes using regex
    # Remove -crosslingual-sts or -sts (order matters - check longer pattern first)
    cleaned = re.sub(r"-crosslingual-sts$", "", cleaned)
    cleaned = re.sub(r"-sts$", "", cleaned)

    # Remove -pairclassification
    cleaned = re.sub(r"-pairclassification$", "", cleaned)

    # Remove -clustering (including -clustering-p2p, -clustering-s2s)
    cleaned = re.sub(r"-clustering(-[ps]2[ps])?$", "", cleaned)

    # Remove -reranking
    cleaned = re.sub(r"-reranking$", "", cleaned)

    # Step 3: Special case - convert "sickr" to "sick-r"
    if cleaned.lower() == "sickr":
        cleaned = "sick-r"

    return cleaned


def save_df(
    df: pd.DataFrame,
    output_path: str | Path,
    file_name: str | None = None,
    file_format: str = "jsonl",
) -> None:
    file_format = file_format.lower()
    if file_name:
        output_path_file = os.path.join(output_path, f"{file_name}.{file_format}")
    else:
        output_path_file = f"{output_path}.{file_format}"
    dir_path = os.path.dirname(output_path_file)

    # Create directory if it doesn't exist
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Overwrite check (auto-overwrite if file contains debug in path)
    if os.path.exists(output_path_file):
        if "debug" in output_path_file.lower():
            print(f"Debug mode: Auto-overwriting existing file '{output_path_file}'")
        else:
            try:
                overwrite = input(f"\nFile '{output_path_file}' already exists. Overwrite? [y/n] ")
                if not overwrite.lower().startswith("y"):
                    print("File not overwritten.")
                    return
            except EOFError:
                print(f"Non-interactive mode: Auto-overwriting existing file '{output_path_file}'")
                pass

    # Save based on file format
    if file_format in ["jsonl", "json"]:
        df.to_json(output_path_file, orient="records", lines=True)
    elif file_format == "csv":
        df.to_csv(output_path_file, index=False)
    elif file_format == "tsv":
        df.to_csv(output_path_file, sep="\t", index=False)
    elif file_format == "xlsx":
        df.to_excel(output_path_file, engine="openpyxl", index=False)
    else:
        print(f"Unsupported file format: {file_format}")
        return

    print(f"[OK] File saved as '{output_path_file}' in {file_format} format.")


def cleanup_model(model: Any, model_name: str | None = None) -> None:
    """
    Clean up a SentenceTransformer model and release GPU memory.

    Args:
        model: The SentenceTransformer model to clean up
        model_name: Optional model name for logging
    """
    if model is not None:
        # Don't move to CPU - just delete the model reference
        # Moving to CPU causes the next task to run on CPU
        del model

    # Force garbage collection
    gc.collect()

    # Clear GPU cache multiple times for thorough cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # Wait a moment and clear cache again
        time.sleep(0.1)
        torch.cuda.empty_cache()

    if model_name:
        print(f"Cleaned up model: {model_name}")


def get_model_name(model: Any) -> str:
    """
    Get the model name from a SentenceTransformer model.

    Args:
        model: SentenceTransformer model

    Returns:
        Simple, hashable model name string
    """
    if hasattr(model, "model_name"):
        return str(model.model_name)
    elif hasattr(model, "_modules") and hasattr(model._modules.get("0", None), "model_name"):
        # Some SentenceTransformer versions store name differently
        return str(model._modules["0"].model_name)
    elif hasattr(model, "model_name_or_path"):
        return str(model.model_name_or_path)
    else:
        # For SentenceTransformer models, try to extract a readable name
        model_str = str(model)
        if "SentenceTransformer" in model_str:
            # Try to find the model path in the transformer config
            if hasattr(model, "_modules") and "0" in model._modules:
                transformer = model._modules["0"]
                if hasattr(transformer, "auto_model") and hasattr(
                    transformer.auto_model, "name_or_path"
                ):
                    return str(transformer.auto_model.name_or_path)
                elif hasattr(transformer, "auto_model") and hasattr(
                    transformer.auto_model.config, "_name_or_path"
                ):
                    return str(transformer.auto_model.config._name_or_path)

            # If we can't extract a clean name, try to parse from string representation
            # Look for model paths like 'sentence-transformers/all-mpnet-base-v2'
            match = re.search(r"sentence-transformers/([a-zA-Z0-9-_]+)", model_str)
            if match:
                return match.group(1)

            # Look for other model paths
            match = re.search(r"([a-zA-Z0-9-_]+/[a-zA-Z0-9-_]+)", model_str)
            if match:
                return match.group(1)

            # If all else fails, return a simplified name
            return "SentenceTransformer"

        # For non-SentenceTransformer models, try to get class name
        return model.__class__.__name__


def is_multilingual_config(config: dict[str, Any], dataset_name: str | None = None) -> bool:
    """
    Check if the configuration specifies multilingual evaluation.

    Args:
        config: Configuration dictionary containing languages setting
        dataset_name: Optional dataset name to check metadata when languages is None

    Returns:
        True if config uses all languages or non-English languages, False for English-only
    """
    # Check if languages key exists in config
    if "languages" not in config:
        # No languages specified → check dataset metadata if available
        if dataset_name:
            try:
                from .dataset_metadata import get_dataset_metadata

                metadata = get_dataset_metadata(dataset_name)
                return metadata.is_non_en()
            except (ValueError, ImportError):
                # If metadata not found or import fails, default to True for backward compat
                return True
        return True

    languages = config["languages"]

    # If languages is None or empty, check dataset metadata if available
    if languages is None or languages == []:
        if dataset_name:
            try:
                from .dataset_metadata import get_dataset_metadata

                metadata = get_dataset_metadata(dataset_name)
                return metadata.is_non_en()
            except (ValueError, ImportError):
                # If metadata not found or import fails, default to True for backward compat
                return True
        return True

    # If languages contains only English variants, not multilingual
    english_variants = {"en", "en-en", "en-ext", "eng", "eng-latn"}
    languages_set = {lang.lower() for lang in languages}

    # If all languages are English variants, not multilingual
    if languages_set.issubset(english_variants):
        return False

    # Otherwise, contains non-English languages, so multilingual
    return True


def should_use_multilingual_suffix(dataset_name: str, config: dict[str, Any]) -> bool:
    """
    Determine if a dataset should use the _multilingual suffix in paraphrase paths.

    Only STS17 and STS22 crosslingual datasets use the multilingual suffix when
    the config includes non-English languages. All other datasets use the
    dataset name as-is without suffix.

    Args:
        dataset_name: Name of the dataset (case-insensitive)
        config: Configuration dictionary containing languages setting

    Returns:
        True if _multilingual suffix should be appended, False otherwise
    """
    # Normalize dataset name to lowercase for comparison
    dataset_lower = dataset_name.lower()

    # Only STS17 and STS22 crosslingual datasets use multilingual suffix
    crosslingual_sts_datasets = {
        "sts17",
        "sts17-crosslingual-sts",
        "sts22",
        "sts22-crosslingual-sts",
    }

    if dataset_lower in crosslingual_sts_datasets:
        return is_multilingual_config(config, dataset_name)

    # All other datasets never use multilingual suffix
    return False


def apply_language_filter(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Apply language filtering to a DataFrame based on configuration.

    Central utility function to handle language filtering consistently across all tasks.
    Supports both paraphrase files (with 'language' column) and MTEB datasets (with 'lang' column).

    Args:
        df: DataFrame to filter
        config: Configuration dictionary containing languages setting

    Returns:
        Filtered DataFrame
    """
    # Determine which language column to use
    lang_col = None
    if "language" in df.columns:
        lang_col = "language"
    elif "lang" in df.columns:
        lang_col = "lang"
    else:
        # No language column, return as-is
        return df

    # Get language filter from config
    languages = config.get("languages", None)

    # If no filtering specified (None or empty), return all languages
    if not languages or len(languages) == 0:
        return df

    # Apply filtering
    original_len = len(df)

    # Map 2-letter ISO 639-1 codes to 3-letter ISO 639-3 codes (used by MTEB datasets)
    iso_639_1_to_639_3 = {
        "en": "eng",  # English
        "ar": "ara",  # Arabic
        "de": "deu",  # German
        "es": "spa",  # Spanish
        "fr": "fra",  # French
        "it": "ita",  # Italian
        "nl": "nld",  # Dutch
        "pl": "pol",  # Polish
        "ru": "rus",  # Russian
        "tr": "tur",  # Turkish
        "zh": "zho",  # Chinese
        "ko": "kor",  # Korean
        "ja": "jpn",  # Japanese
        # Add more Masakhane languages (African languages)
        "am": "amh",  # Amharic
        "ha": "hau",  # Hausa
        "ig": "ibo",  # Igbo
        "om": "orm",  # Oromo
        "rw": "kin",  # Kinyarwanda
        "so": "som",  # Somali
        "sw": "swa",  # Swahili
        "ti": "tir",  # Tigrinya
        "yo": "yor",  # Yoruba
    }

    # Normalize language codes to their base forms (3-letter ISO 639-3 codes)
    normalized_langs = set()
    for lang in languages:
        lang_lower = str(lang).lower()

        # Extract base language code (handle both "en" and "en-en" formats)
        if "-" in lang_lower:
            parts = lang_lower.split("-")
            for part in parts:
                # Map 2-letter to 3-letter if needed
                if part in iso_639_1_to_639_3:
                    normalized_langs.add(iso_639_1_to_639_3[part])
                else:
                    normalized_langs.add(part)
        else:
            # Map 2-letter to 3-letter if needed
            if lang_lower in iso_639_1_to_639_3:
                normalized_langs.add(iso_639_1_to_639_3[lang_lower])
            else:
                normalized_langs.add(lang_lower)

    # Build reverse mapping (3-letter -> 2-letter) for dataset compatibility
    # Some datasets use 2-letter codes (en-en) while others use 3-letter (eng-eng)
    iso_639_3_to_639_1 = {v: k for k, v in iso_639_1_to_639_3.items()}

    # Build filtered language set
    # Include:
    # 1. Base language codes (eng, spa, fra) AND their 2-letter equivalents (en, es, fr)
    # 2. Monolingual pairs (eng-eng, spa-spa) AND their 2-letter equivalents (en-en, es-es)
    # 3. Crosslingual pairs between any combination of specified languages (both 2 and 3 letter)
    filtered_languages = set()

    for lang in normalized_langs:
        # Add base language code (e.g., "eng", "spa")
        filtered_languages.add(lang)

        # Add 2-letter equivalent if it exists (e.g., "en" for "eng")
        if lang in iso_639_3_to_639_1:
            filtered_languages.add(iso_639_3_to_639_1[lang])

        # Add monolingual pair (e.g., "eng-eng", "spa-spa")
        filtered_languages.add(f"{lang}-{lang}")

        # Add 2-letter monolingual pair (e.g., "en-en", "es-es")
        if lang in iso_639_3_to_639_1:
            lang_2letter = iso_639_3_to_639_1[lang]
            filtered_languages.add(f"{lang_2letter}-{lang_2letter}")

    # Add all crosslingual pairs (both directions) between specified languages
    lang_list = sorted(list(normalized_langs))
    for i, lang1 in enumerate(lang_list):
        for lang2 in lang_list[i+1:]:  # Only pairs, not duplicates
            # Add 3-letter pairs
            filtered_languages.add(f"{lang1}-{lang2}")
            filtered_languages.add(f"{lang2}-{lang1}")

            # Add 2-letter pairs if both languages have 2-letter codes
            if lang1 in iso_639_3_to_639_1 and lang2 in iso_639_3_to_639_1:
                lang1_2letter = iso_639_3_to_639_1[lang1]
                lang2_2letter = iso_639_3_to_639_1[lang2]
                filtered_languages.add(f"{lang1_2letter}-{lang2_2letter}")
                filtered_languages.add(f"{lang2_2letter}-{lang1_2letter}")

    # Filter DataFrame
    df_filtered = df[df[lang_col].str.lower().isin(filtered_languages)].copy()

    if len(df_filtered) < original_len:
        print(
            f"Language filtering: {original_len} -> {len(df_filtered)} samples (languages: {sorted(list(filtered_languages))})"
        )

    return df_filtered


def get_paraphrasing_seed(config: dict[str, Any]) -> int | list[int]:
    """
    Get the seed to use for paraphrasing operations.

    Args:
        config: Configuration dictionary

    Returns:
        Seed value for paraphrasing - can be int or list of ints
    """
    seed_value = config.get("seed_paraphrasing", config.get("seed", 1337))

    # Handle both int and list types
    if isinstance(seed_value, list):
        return seed_value
    else:
        return int(seed_value)


def play_completion_sound() -> None:
    """
    Play a sound notification on Linux when evaluation completes.

    Tries multiple methods in order:
    1. pw-play (PipeWire)
    2. paplay (PulseAudio)
    3. aplay (ALSA) - converts oga to wav first
    4. Terminal bell character
    5. Silent fallback
    """
    import subprocess

    sound_file = "/usr/share/sounds/freedesktop/stereo/complete.oga"

    # Try pw-play (PipeWire)
    try:
        result = subprocess.run(
            ["pw-play", sound_file],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        if result.returncode == 0:
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try paplay (PulseAudio)
    try:
        result = subprocess.run(
            ["paplay", sound_file],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        if result.returncode == 0:
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try aplay (ALSA only supports wav, so convert oga with ffmpeg)
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", sound_file, "-f", "wav", "pipe:1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        if result.returncode == 0:
            subprocess.run(
                ["aplay", "-q"],
                input=result.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback to terminal bell
    print("\a")
