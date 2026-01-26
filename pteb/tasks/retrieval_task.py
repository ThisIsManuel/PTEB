"""
Retrieval task implementation for PTEB.

This module implements the RetrievalTask class that handles retrieval dataset loading,
paraphrase generation/loading, and evaluation with retrieval metrics including nDCG@10.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..evaluators.retrieval_evaluator import RetrievalEvaluator
from ..utils import cleanup_model, should_use_multilingual_suffix
from .paraphrase_task import ParaphraseTask


class RetrievalTask(ParaphraseTask):
    """
    Retrieval task with paraphrase-enhanced evaluation.

    This class handles:
    - Loading retrieval datasets from HuggingFace Hub (corpus, queries, qrels)
    - Generating or loading paraphrases for queries
    - Evaluating embedding models on original and paraphrased queries
    - Computing retrieval metrics with nDCG@10 as the primary metric
    """

    # Whether to ignore identical IDs for specific datasets
    # Other prompt information is now centralized in para_mteb.prompts
    IGNORE_IDENTICAL_IDS_DICT = {
        "arguana": True,
        "TwitterHjerneRetrieval": False,
        "LegalQuAD": False,
    }

    def __init__(self, datasets: List[str], config: Dict[str, Any], **kwargs):
        """
        Initialize the Retrieval task.

        Args:
            datasets: List of retrieval dataset names to evaluate on
            config: Configuration dictionary from CLI arguments
            **kwargs: Additional parameters
        """
        # Validate that all datasets have defined ignore_identical_ids settings
        # Note: Retrieval tasks rely on model-specific prompts (e.g., EmbeddingGemma),
        # not hardcoded task prompts, so we skip prompt validation here
        for dataset in datasets:
            # Validate retrieval-specific settings
            if dataset not in self.IGNORE_IDENTICAL_IDS_DICT:
                raise NotImplementedError(
                    f"Dataset '{dataset}' does not have a defined ignore_identical_ids setting. "
                    f"Available datasets with settings: {list(self.IGNORE_IDENTICAL_IDS_DICT.keys())}"
                )

        super().__init__(
            task_name="Retrieval",
            task_description="Retrieval evaluation with paraphrase augmentation using nDCG@10 as primary metric",
            datasets=datasets,
            config=config,
            **kwargs,
        )

        # Dataset-specific settings are now handled by the centralized prompt system
        ignore_identical_ids = {
            ds: self.IGNORE_IDENTICAL_IDS_DICT[ds] for ds in datasets
        }
        ignore_identical_ids = set(ignore_identical_ids.values())
        if len(ignore_identical_ids) > 1:
            raise ValueError(
                f"All datasets must have the same ignore_identical_ids setting for retrieval tasks. "
                f"Found: {ignore_identical_ids}"
            )
        self.ignore_identical_ids = ignore_identical_ids.pop()
        # Retrieval-specific configuration
        self.score_function = config.get("score_function", "cos_sim")
        self.corpus_chunk_size = config.get("corpus_chunk_size")

        # For now, use the default ignore_identical_ids from config
        self.evaluator = RetrievalEvaluator(
            ignore_identical_ids=self.ignore_identical_ids,
            corpus_chunk_size=self.corpus_chunk_size,
            seed=self.seed_general,
        )

        print(
            f"Initialized Retrieval task with primary metric: {self.evaluator.main_metric}"
        )
        if self.corpus_chunk_size:
            print(
                f"Corpus chunking enabled: {self.corpus_chunk_size} documents per chunk"
            )
        else:
            print("Corpus chunking disabled - encoding all documents at once")

    def downsample_data(
        self,
        queries: dict[str, str],
        qrels: dict[str, dict[str, int]],
    ) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
        """Downsample queries while preserving corpus integrity."""
        downsample = self.config.get("downsample")
        if downsample is None or downsample >= 1.0:
            return queries, qrels

        import random

        rng = random.Random(self.seed_general)

        n_sample = max(1, int(len(queries) * downsample))
        sampled_ids = set(rng.sample(list(queries.keys()), n_sample))

        sampled_queries = {qid: queries[qid] for qid in sampled_ids}
        filtered_qrels = {qid: qrels[qid] for qid in sampled_ids if qid in qrels}

        print(f"Downsampled to {len(sampled_queries)} queries ({downsample*100:.1f}%)")
        return sampled_queries, filtered_qrels

    def load_hf_data(
        self, dataset_name: str
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        """
        Load and return the specified retrieval dataset from HuggingFace Hub.

        Args:
            dataset_name: Name of the retrieval dataset to load

        Returns:
            Tuple of (corpus, queries, qrels) where:
            - corpus: Dict mapping doc_id -> {'title': str, 'text': str}
            - queries: Dict mapping query_id -> query_text
            - qrels: Dict mapping query_id -> {doc_id -> relevance_score}

        Raises:
            ValueError: If dataset_name is not supported
            FileNotFoundError: If dataset cannot be loaded
        """

        # Get dataset-specific split and loading config from metadata
        metadata = self.dataset_metadata.get(self.get_canonical_name(dataset_name))
        dataset_split = (
            metadata.eval_split
            if metadata and hasattr(metadata, "eval_split")
            else self.split
        )
        loading_config = (
            metadata.loading_config
            if metadata and hasattr(metadata, "loading_config")
            else None
        )

        # Extract split names from loading_config if available
        corpus_split = None
        queries_split = None
        qrels_split = None
        if loading_config:
            corpus_split = loading_config.get("corpus_split", "corpus")
            queries_split = loading_config.get("queries_split", "queries")
            qrels_split = (
                loading_config.get("qrels_split")
                or loading_config.get("default_split")
                or dataset_split
            )

        # Get HuggingFace dataset name from metadata (handles alternate names)
        metadata = self.dataset_metadata[self.get_canonical_name(dataset_name)]
        hf_dataset_name = metadata.hf_dataset_name


        print(
            f"Loading retrieval dataset {hf_dataset_name} from HuggingFace Hub..."
        )

        # Import here to avoid circular imports
        from datasets import load_dataset

        # Load different parts of the dataset
        try:
            # Try loading corpus, queries, and qrels separately (MTEB format)
            # Use loading_config if available, otherwise fall back to defaults
            # Pattern 1: Corpus and queries have self-named splits (corpus→corpus, queries→queries)
            # Pattern 2: All configs use the data split name (train/test)
            if corpus_split:
                corpus_data = load_dataset(
                    hf_dataset_name, name="corpus", split=corpus_split
                )
            else:
                try:
                    corpus_data = load_dataset(
                        hf_dataset_name, name="corpus", split="corpus"
                    )
                except Exception:
                    corpus_data = load_dataset(
                        hf_dataset_name, name="corpus", split=self.split
                    )

            if queries_split:
                queries_data = load_dataset(
                    hf_dataset_name, name="queries", split=queries_split
                )
            else:
                try:
                    queries_data = load_dataset(
                        hf_dataset_name, name="queries", split="queries"
                    )
                except Exception:
                    queries_data = load_dataset(
                        hf_dataset_name, name="queries", split=self.split
                    )

            if qrels_split:
                try:
                    qrels_data = load_dataset(
                        hf_dataset_name, name="qrels", split=qrels_split
                    )
                except Exception:
                    qrels_data = load_dataset(
                        hf_dataset_name, name="default", split=qrels_split
                    )
            else:
                try:
                    qrels_data = load_dataset(
                        hf_dataset_name, name="qrels", split=self.split
                    )
                except Exception:
                    qrels_data = load_dataset(
                        hf_dataset_name, name="default", split=self.split
                    )

            print(f"Loaded corpus: {len(corpus_data)} documents")
            print(f"Loaded queries: {len(queries_data)} queries")
            print(f"Loaded qrels: {len(qrels_data)} relevance judgments")

            # Convert corpus
            corpus = {}
            for doc in corpus_data:
                doc_id = doc.get("_id", doc.get("id"))
                corpus[doc_id] = {
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                }

            # Convert queries
            queries = {}
            for query in queries_data:
                query_id = query.get("_id", query.get("id"))
                queries[query_id] = query.get("text", "")

            # Convert qrels
            qrels = {}
            for rel in qrels_data:
                query_id = str(
                    rel.get("query-id", rel.get("query_id", rel.get("qid")))
                )
                doc_id = str(
                    rel.get("corpus-id", rel.get("doc_id", rel.get("pid")))
                )
                relevance = int(rel.get("score", rel.get("relevance", 1)))

                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = relevance

        except Exception as e:
            print(
                f"Could not load in MTEB format ({e}), trying alternative format..."
            )

            hf_dataset = load_dataset(hf_dataset_name, split=dataset_split)

            # Try alternative format where each example is a query-document pair
            corpus_dict = {}
            queries_dict = {}
            qrels_dict = {}

            for example in hf_dataset:
                # Handle different possible field names
                query_id = example.get(
                    "query-id", example.get("query_id", example.get("qid"))
                )
                query_text = example.get("query", example.get("query_text"))
                doc_id = example.get(
                    "corpus-id", example.get("doc_id", example.get("pid"))
                )
                doc_title = example.get("title", "")
                doc_text = example.get("text", example.get("passage", ""))
                relevance = example.get("score", example.get("relevance", 1))

                if query_id and query_text:
                    queries_dict[str(query_id)] = query_text

                if doc_id and doc_text:
                    corpus_dict[str(doc_id)] = {
                        "title": doc_title,
                        "text": doc_text,
                    }

                if query_id and doc_id and relevance > 0:
                    if str(query_id) not in qrels_dict:
                        qrels_dict[str(query_id)] = {}
                    qrels_dict[str(query_id)][str(doc_id)] = int(relevance)

            corpus = corpus_dict
            queries = queries_dict
            qrels = qrels_dict

        print(
            f"Loaded {len(corpus)} documents, {len(queries)} queries, {len(qrels)} qrels"
        )

        # Apply downsampling if configured
        queries, qrels = self.downsample_data(queries, qrels)

        return corpus, queries, qrels

    def evaluate(
        self,
        dataset_name: str,
        embedding_model: Any,
        emb_model_name: str,
        data: pd.DataFrame | None = None,
        gen_model_name: str | None = None,
        gen_api_name: str | None = None,
        pregenerated_paraphrases: dict[str, Any] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate the embedding model on retrieval data with paraphrases.

        Args:
            dataset_name: Name of the dataset being evaluated
            embedding_model: The embedding model to evaluate
            data: DataFrame with the dataset (if None, will load from dataset_name)
            gen_model_name: Name of generative model for paraphrases
            gen_api_name: API name for generative model
            pregenerated_paraphrases: Pre-generated paraphrases for queries
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation results with nDCG@10 as primary metric
        """
        # Load retrieval data if not provided
        if data is None:
            corpus, queries, qrels = self.load_hf_data(dataset_name)
        else:
            # Convert DataFrame back to structured format
            corpus = {}
            queries = {}
            qrels = {}

            for _, row in data.iterrows():
                # Extract unique queries
                if row["query_id"] not in queries:
                    queries[row["query_id"]] = row["query_text"]

                # Extract unique documents
                if row["doc_id"] not in corpus:
                    corpus[row["doc_id"]] = {
                        "title": row["doc_title"],
                        "text": row["doc_text"],
                    }

                # Build qrels
                if row["query_id"] not in qrels:
                    qrels[row["query_id"]] = {}
                qrels[row["query_id"]][row["doc_id"]] = row["relevance"]

        # Handle paraphrased queries if available
        if self.gen_model and pregenerated_paraphrases:
            print(
                f"Processing with pre-generated query paraphrases for {gen_model_name}..."
            )

            # Use paraphrased queries
            paraphrased_queries = {}
            if "paraphrases1" in pregenerated_paraphrases:
                # Map paraphrased queries back to query IDs
                original_queries = pregenerated_paraphrases.get("original1", [])
                paraphrased_texts = pregenerated_paraphrases.get("paraphrases1", [])

                # Create mapping from original to paraphrased
                for i, (orig_query, para_query) in enumerate(
                    zip(original_queries, paraphrased_texts)
                ):
                    # Find query ID for this original query
                    for qid, qtext in queries.items():
                        if qtext == orig_query:
                            paraphrased_queries[qid] = para_query
                            break

            # Use paraphrased queries if available, otherwise use original
            eval_queries = paraphrased_queries if paraphrased_queries else queries

            # Evaluate with paraphrased queries
            result = self.evaluator(
                embedding_model,
                corpus=corpus,
                queries=eval_queries,
                qrels=qrels,
                score_function=self.score_function,
                encode_kwargs=kwargs.get("encode_kwargs", {}),
                task_name="retrieval",
                dataset_name=dataset_name,
            )

            # Add comparison with original queries if paraphrases were used
            if paraphrased_queries:
                original_result = self.evaluator(
                    embedding_model,
                    corpus=corpus,
                    queries=queries,
                    qrels=qrels,
                    score_function=self.score_function,
                    encode_kwargs=kwargs.get("encode_kwargs", {}),
                    task_name="retrieval",
                    dataset_name=dataset_name,
                )

                # Combine results with original_ and paraphrased_ prefixes
                combined_result = {}
                for metric, value in original_result.items():
                    combined_result[f"original_{metric}"] = value
                for metric, value in result.items():
                    combined_result[f"paraphrased_{metric}"] = value

                # Add standardized main score columns (NDCG@10 for retrieval)
                main_metric = "NDCG@10"
                combined_result["original_main_score"] = original_result.get(
                    main_metric, 0.0
                )
                combined_result["paraphrased_main_score"] = result.get(main_metric, 0.0)

                result = combined_result

            # Add model information
            result.update(
                {
                    "emb_model": emb_model_name,
                    "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
                    "model_max_seq_length": kwargs.get(
                        "model_max_seq_length",
                        getattr(embedding_model, "max_seq_length", None),
                    ),
                    "emb_batch_size": self.config.get("emb_batch_size", 32),
                    "dtype": str(self.config.get("st_dtype", "torch.float32")),
                    "encoding_chunk_size": self.config.get("encoding_chunk_size", None),
                    "corpus_chunk_size": self.config.get("corpus_chunk_size", None),
                    "emb_model_api": self.config.get(
                        "emb_model_api", "sentence-transformers"
                    ),
                    "gen_model": gen_model_name,
                    "gen_api": gen_api_name,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    # Missing fields for consistency
                    "timestamp": self.timestamp,
                    "run": kwargs.get("run", 0),
                    "embedding_generation_time": kwargs.get(
                        "embedding_generation_time", None
                    ),
                }
            )

            return result

        elif self.gen_model:
            # Generate paraphrases on-demand (shouldn't happen in normal flow)
            print(
                f"Warning: No pre-generated paraphrases found, generating on-demand for {gen_model_name}..."
            )

            # Extract queries for paraphrase generation
            query_texts = list(queries.values())

            # Generate paraphrases for queries using the new cleaner method
            paraphrased_queries = self._generate_paraphrases(
                texts=query_texts,
                gen_model_name=gen_model_name,
                gen_api_name=gen_api_name,
                dataset_name=dataset_name,
            )

            # Save paraphrases if requested
            if self.save_paraphrases:
                self._save_paraphrases(
                    query_ids=list(queries.keys()),
                    original_queries=query_texts,
                    paraphrased_queries=paraphrased_queries,
                    gen_model_name=gen_model_name,
                    gen_api_name=gen_api_name,
                    dataset_name=dataset_name,
                )

            # Map paraphrased queries back to query IDs
            paraphrased_query_dict = {}
            for i, (qid, paraphrased_text) in enumerate(
                zip(queries.keys(), paraphrased_queries)
            ):
                paraphrased_query_dict[qid] = paraphrased_text

            # Evaluate both original and paraphrased
            original_result = self.evaluator(
                embedding_model,
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                score_function=self.score_function,
                encode_kwargs=kwargs.get("encode_kwargs", {}),
                task_name="retrieval",
                dataset_name=dataset_name,
            )

            paraphrased_result = self.evaluator(
                embedding_model,
                corpus=corpus,
                queries=paraphrased_query_dict,
                qrels=qrels,
                score_function=self.score_function,
                encode_kwargs=kwargs.get("encode_kwargs", {}),
                task_name="retrieval",
                dataset_name=dataset_name,
            )

            # Combine results
            result = {}
            for metric, value in original_result.items():
                result[f"original_{metric}"] = value
            for metric, value in paraphrased_result.items():
                result[f"paraphrased_{metric}"] = value

            # Add standardized main score columns (NDCG@10 for retrieval)
            main_metric = "NDCG@10"
            result["original_main_score"] = original_result.get(main_metric, 0.0)
            result["paraphrased_main_score"] = paraphrased_result.get(main_metric, 0.0)

            # Add model information
            result.update(
                {
                    "emb_model": emb_model_name,
                    "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
                    "model_max_seq_length": kwargs.get(
                        "model_max_seq_length",
                        getattr(embedding_model, "max_seq_length", None),
                    ),
                    "emb_batch_size": self.config.get("emb_batch_size", 32),
                    "dtype": str(self.config.get("st_dtype", "torch.float32")),
                    "encoding_chunk_size": self.config.get("encoding_chunk_size", None),
                    "corpus_chunk_size": self.config.get("corpus_chunk_size", None),
                    "emb_model_api": self.config.get(
                        "emb_model_api", "sentence-transformers"
                    ),
                    "gen_model": gen_model_name,
                    "gen_api": gen_api_name,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    # Missing fields for consistency
                    "timestamp": self.timestamp,
                    "run": kwargs.get("run", 0),
                    "embedding_generation_time": kwargs.get(
                        "embedding_generation_time", None
                    ),
                }
            )

            return result

        else:
            # No generative models - evaluate only original queries
            individual_eval_start = time.time()
            result = self.evaluator(
                embedding_model,
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                score_function=self.score_function,
                encode_kwargs=kwargs.get("encode_kwargs", {}),
                task_name="retrieval",
                dataset_name=dataset_name,
            )
            individual_eval_time = time.time() - individual_eval_start

            # Add model information and timing
            result.update(
                {
                    "emb_model": emb_model_name,
                    "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
                    "model_max_seq_length": kwargs.get(
                        "model_max_seq_length",
                        getattr(embedding_model, "max_seq_length", None),
                    ),
                    "emb_batch_size": self.config.get("emb_batch_size", 32),
                    "dtype": str(self.config.get("st_dtype", "torch.float32")),
                    "encoding_chunk_size": self.config.get("encoding_chunk_size", None),
                    "corpus_chunk_size": self.config.get("corpus_chunk_size", None),
                    "emb_model_api": self.config.get(
                        "emb_model_api", "sentence-transformers"
                    ),
                    "gen_model": None,
                    "gen_api": None,
                    "individual_eval_time": individual_eval_time,
                    # Missing fields for consistency
                    "timestamp": self.timestamp,
                    "run": kwargs.get("run", 0),
                    "temperature": None,
                    "top_p": None,
                    "embedding_generation_time": kwargs.get(
                        "embedding_generation_time", None
                    ),
                }
            )

            return result

    def run_full_evaluation(
        self, embedding_models: List[Any], model_names: List[str], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run complete retrieval evaluation across all datasets and embedding models.

        Args:
            embedding_models: List of embedding models to evaluate
            model_names: List of model name strings corresponding to embedding_models
            **kwargs: Additional evaluation parameters

        Returns:
            List of all evaluation results
        """
        all_results = []

        for dataset_name in self.datasets:
            print(f"\n{'=' * 50}")
            print(f"Evaluating dataset: {dataset_name}")
            print(f"{'=' * 50}")

            try:
                # Load retrieval data
                corpus, queries, qrels = self.load_hf_data(dataset_name)

                # Apply debug sampling if needed (already handled in load_data)
                print(
                    f"Dataset loaded: {len(corpus)} documents, {len(queries)} queries"
                )

                # Evaluate with each embedding model
                for model_idx, emb_model in enumerate(embedding_models):
                    emb_model_name = model_names[model_idx]
                    print(f"\nEvaluating with embedding model: {emb_model_name}")

                    if self.gen_model:
                        # Evaluate with paraphrases using Type 1 architecture (run loop)
                        gen_api = self.gen_model_api

                        # Add run loop for Type 1 architecture
                        for run in range(self.n_runs):
                            run_seed = self.get_seed_for_run(run)
                            print(
                                f"  With paraphrases from: {self.gen_model} ({gen_api}) - Run {run + 1}/{self.n_runs} (seed: {run_seed})"
                            )

                            eval_start = time.time()

                            # Generate paraphrases for this specific run
                            original_queries = queries
                            query_texts = list(queries.values())

                            # Generate paraphrases for this run
                            paraphrased_query_list = self._generate_paraphrases(
                                texts=query_texts,
                                gen_model_name=self.gen_model,
                                gen_api_name=gen_api,
                                dataset_name=dataset_name,
                                seed=run_seed,
                            )

                            # Map paraphrased queries back to query IDs
                            paraphrased_queries = {}
                            for qid, paraphrased_text in zip(
                                queries.keys(), paraphrased_query_list
                            ):
                                paraphrased_queries[qid] = paraphrased_text

                            # Save paraphrases if requested
                            if self.save_paraphrases:
                                seeds_used = [run_seed] * len(query_texts)
                                self._save_paraphrases(
                                    query_ids=list(queries.keys()),
                                    original_queries=query_texts,
                                    paraphrased_queries=paraphrased_query_list,
                                    gen_model_name=self.gen_model,
                                    gen_api_name=gen_api,
                                    dataset_name=dataset_name,
                                    seeds_used=seeds_used,
                                )

                            # Evaluate both original and paraphrased
                            original_result = self.evaluator(
                                emb_model,
                                corpus=corpus,
                                queries=original_queries,
                                qrels=qrels,
                                score_function=self.score_function,
                                encode_kwargs=kwargs.get("encode_kwargs", {}),
                                task_name="retrieval",
                                dataset_name=dataset_name,
                            )

                            paraphrased_result = self.evaluator(
                                emb_model,
                                corpus=corpus,
                                queries=paraphrased_queries,
                                qrels=qrels,
                                score_function=self.score_function,
                                encode_kwargs=kwargs.get("encode_kwargs", {}),
                                task_name="retrieval",
                                dataset_name=dataset_name,
                            )

                            eval_time = time.time() - eval_start

                            # Combine results with original_ and paraphrased_ prefixes
                            result = {}
                            for metric, value in original_result.items():
                                result[f"original_{metric}"] = value
                            for metric, value in paraphrased_result.items():
                                result[f"paraphrased_{metric}"] = value

                            # Add standardized main score columns (NDCG@10 for retrieval)
                            main_metric = "NDCG@10"
                            result["original_main_score"] = original_result.get(
                                main_metric, 0.0
                            )
                            result["paraphrased_main_score"] = paraphrased_result.get(
                                main_metric, 0.0
                            )

                            # Add metadata to result
                            result.update(
                                {
                                    "timestamp": self.timestamp,
                                    "task": self.task_name,
                                    "dataset": self.get_canonical_name(dataset_name),
                                    "n_samples": len(queries),
                                    "runtime": eval_time,
                                    "seed_general": self.seed_general,
                                    "seed_paraphrasing": run_seed,  # Use run-specific seed (already incremented)
                                    "run": run + 1,  # Add run information
                                    "emb_model": emb_model_name,
                                    "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
                                    "model_max_seq_length": kwargs.get(
                                        "model_max_seq_length",
                                        getattr(emb_model, "max_seq_length", None),
                                    ),
                                    "emb_batch_size": self.config.get(
                                        "emb_batch_size", 32
                                    ),
                                    "dtype": str(
                                        self.config.get("st_dtype", "torch.float32")
                                    ),
                                    "encoding_chunk_size": self.config.get(
                                        "encoding_chunk_size", None
                                    ),
                                    "corpus_chunk_size": self.config.get(
                                        "corpus_chunk_size", None
                                    ),
                                    "emb_model_api": self.config.get(
                                        "emb_model_api",
                                        "sentence-transformers",
                                    ),
                                    "gen_model": self.gen_model,
                                    "gen_api": gen_api,
                                    "paraphrase_generation_time": eval_time,
                                    "temperature": self.temperature,
                                    "top_p": self.top_p,
                                }
                            )

                            all_results.append(result)

                            # Show both original and paraphrased main metrics
                            orig_metric = result.get(
                                f"original_{self.evaluator.main_metric}", "N/A"
                            )
                            para_metric = result.get(
                                f"paraphrased_{self.evaluator.main_metric}",
                                "N/A",
                            )
                            print(
                                f"      Original {self.evaluator.main_metric}: {orig_metric:.4f}"
                            )
                            print(
                                f"      Paraphrased {self.evaluator.main_metric}: {para_metric:.4f}"
                            )

                            # Display main scores
                            if (
                                "original_main_score" in result
                                and "paraphrased_main_score" in result
                            ):
                                print(
                                    f"    Main Score - Original: {result['original_main_score']:.4f}, Paraphrased: {result['paraphrased_main_score']:.4f}"
                                )

                            print(f"    Completed in {eval_time:.2f} seconds")
                    else:
                        # No generative models - evaluate only original queries
                        eval_start = time.time()

                        result = self.evaluator(
                            emb_model,
                            corpus=corpus,
                            queries=queries,
                            qrels=qrels,
                            score_function=self.score_function,
                            encode_kwargs=kwargs.get("encode_kwargs", {}),
                            task_name="retrieval",
                            dataset_name=dataset_name,
                        )

                        eval_time = time.time() - eval_start

                        # Add metadata to result
                        result.update(
                            {
                                "timestamp": self.timestamp,
                                "task": self.task_name,
                                "dataset": self.get_canonical_name(dataset_name),
                                "n_samples": len(queries),
                                "runtime": eval_time,
                                "seed_general": self.seed_general,
                                "seed_paraphrasing": self.seed_paraphrasing,
                                "emb_model": emb_model_name,
                                "model_vram_gb": kwargs.get("model_vram_gb", 0.0),
                                "model_max_seq_length": kwargs.get(
                                    "model_max_seq_length",
                                    getattr(emb_model, "max_seq_length", None),
                                ),
                                "emb_batch_size": self.config.get("emb_batch_size", 32),
                                "dtype": str(
                                    self.config.get("st_dtype", "torch.float32")
                                ),
                                "encoding_chunk_size": self.config.get(
                                    "encoding_chunk_size", None
                                ),
                                "corpus_chunk_size": self.config.get(
                                    "corpus_chunk_size", None
                                ),
                                "emb_model_api": self.config.get(
                                    "emb_model_api", "sentence-transformers"
                                ),
                                "gen_model": None,
                                "gen_api": None,
                                "paraphrase_generation_time": None,
                                # Missing fields for consistency
                                "run": kwargs.get("run", 0),
                                "temperature": None,
                                "top_p": None,
                                "embedding_generation_time": kwargs.get(
                                    "embedding_generation_time", None
                                ),
                            }
                        )

                        all_results.append(result)

                        print(
                            f" {self.evaluator.main_metric}: {result.get(self.evaluator.main_metric, 'N/A'):.4f}"
                        )

                        # Display main score
                        if self.evaluator.main_metric in result:
                            print(
                                f"  Main Score: {result[self.evaluator.main_metric]:.4f}"
                            )

                        print(f"Completed in {eval_time:.2f} seconds")

                    # Clean up embedding model after evaluation to free GPU memory
                    cleanup_model(emb_model, emb_model_name)

            except Exception as e:
                print(f"Error evaluating {dataset_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        total_time = time.time() - self.start_time if self.start_time else 0
        print(f"\nTotal evaluation time: {total_time:.2f} seconds")

        return all_results

    def get_summary_table(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create a summary table from evaluation results showing only NDCG@10.

        Args:
            results: List of evaluation result dictionaries

        Returns:
            DataFrame with key metrics for display, showing only NDCG@10
        """
        if not results:
            return pd.DataFrame()

        # Select key columns for summary, showing only NDCG@10
        summary_cols = [
            "dataset",
            "emb_model",
            "gen_model",
            "NDCG@10",  # Only metric now
            "n_samples",
            "runtime",
        ]

        # Handle paraphrased results
        if any("original_NDCG@10" in result for result in results):
            summary_cols = [
                "dataset",
                "emb_model",
                "gen_model",
                "original_NDCG@10",
                "paraphrased_NDCG@10",
                "n_samples",
                "runtime",
            ]

        # Filter available columns
        available_cols = [col for col in summary_cols if col in results[0]]

        df = pd.DataFrame(results)

        return df[available_cols] if available_cols else df

    def _save_paraphrases(
        self,
        query_ids: List[str],
        original_queries: List[str],
        paraphrased_queries: List[str],
        gen_model_name: str,
        gen_api_name: str,
        dataset_name: str,
        seeds_used: Optional[List[Optional[int]]] = None,
        split_override: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ) -> None:
        """
        Save generated paraphrases to disk with retrieval-specific format.

        Args:
            query_ids: List of query IDs
            original_queries: Original query texts
            paraphrased_queries: Paraphrased query texts
            gen_model_name: Name of the generative model
            gen_api_name: API name for the generative model
            dataset_name: Name of the dataset being processed
            seeds_used: Optional list of seeds used for each paraphrase
            split_override: Override the split name used in the file path
            languages: Optional list of language codes for each query
        """
        import os
        import re

        from ..utils import save_df

        # Create DataFrame with retrieval-specific column names
        paraphrase_df = pd.DataFrame(
            {
                "query_id": query_ids,
                "original_query": original_queries,
                "paraphrase_query": paraphrased_queries,
            }
        )

        # Add language information
        if languages:
            paraphrase_df["language"] = languages
        else:
            # Get language from dataset metadata
            metadata = self.dataset_metadata.get(self.get_canonical_name(dataset_name))
            if metadata and metadata.languages:
                # Use first language from metadata as default
                default_lang = (
                    metadata.languages[0]
                    if isinstance(metadata.languages, list)
                    else metadata.languages
                )
            else:
                # Fallback to English
                default_lang = "eng"
            paraphrase_df["language"] = [default_lang] * len(query_ids)

        # Don't add run number column - in Type 1 architecture each file is a separate run
        # The run information is encoded in the filename instead

        # Determine output directory
        effective_split = split_override if split_override is not None else self.split

        # Use canonical name for path, add multilingual suffix if needed (only for STS17/STS22)
        dataset_name_for_path = self.get_canonical_name(dataset_name)
        if should_use_multilingual_suffix(dataset_name_for_path, self.config):
            dataset_name_for_path = f"{dataset_name_for_path}_multilingual"

        para_output_dir = os.path.join(
            "data",
            "paraphrases",
            "retrieval",
            dataset_name_for_path,
            effective_split,
            re.sub(r":", "-", gen_model_name),
        )

        os.makedirs(para_output_dir, exist_ok=True)

        # Create filename with timestamp
        # Create filename with new seed-run pattern
        if should_use_multilingual_suffix(dataset_name, self.config):
            para_filename = f"{self.timestamp}_{dataset_name}_{gen_model_name}_paraphrases_multilingual"
        else:
            para_filename = (
                f"{self.timestamp}_{dataset_name}_{gen_model_name}_paraphrases"
            )

        # Add seed and run number - use provided seeds_used or default to task seed
        current_seed = (
            seeds_used[0]
            if seeds_used and len(seeds_used) > 0
            else self.seed_paraphrasing
        )
        if self.n_runs > 1:
            # Determine run number from seed
            if current_seed != self.seed_paraphrasing:
                run_offset = self.get_run_index_for_seed(current_seed)
                run_number = run_offset + 1
            else:
                run_number = 1
            para_filename += f"_seed-{current_seed:04d}_run-{run_number}"
        else:
            # Single run case
            para_filename += f"_seed-{current_seed:04d}_run-1"
        para_filename = re.sub(
            r":", "-", para_filename
        )  # Sanitize model name in filename

        # Use JSONL format for consistency
        output_file = os.path.join(para_output_dir, para_filename)

        # Save using the utility function
        save_df(paraphrase_df, output_file, file_format="jsonl")

        print(f"Saved retrieval paraphrases to {output_file}")
