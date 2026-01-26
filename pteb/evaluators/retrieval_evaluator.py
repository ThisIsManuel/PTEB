"""
Retrieval evaluator for PTEB.

This module implements the RetrievalEvaluator class that computes retrieval metrics
including nDCG@10 as the primary metric, following MTEB's evaluation standards.
"""

from typing import Any

import numpy as np
import pytrec_eval

try:
    # Current MTEB location
    from mteb.abstasks.AbsTaskRetrieval import PromptType  # type: ignore[attr-defined]
except Exception:
    try:
        # Older releases occasionally exposed it here
        from mteb.abstasks.AbsTaskReranking import PromptType  # type: ignore[attr-defined]
    except Exception:
        # Final fallback
        class PromptType:
            query = "query"
            document = "document"


from .base_evaluator import BaseEvaluator


class RetrievalEvaluator(BaseEvaluator):
    """
    Evaluator for retrieval tasks using nDCG@10 as the only metric.
    This evaluator computes only nDCG@10 for streamlined retrieval evaluation,
    following MTEB standards.
    """

    def __init__(
        self,
        ignore_identical_ids: bool = False,
        corpus_chunk_size: int | None = None,
        **kwargs,
    ):
        """
        Initialize the retrieval evaluator.

        Args:
            k_values: List of k values for evaluation metrics (defaults to [10] for NDCG@10 only)
            ignore_identical_ids: Whether to ignore identical query and document IDs
            corpus_chunk_size: Number of documents to encode at once (None to disable chunking)
            **kwargs: Additional parameters passed to BaseEvaluator
        """
        super().__init__(**kwargs)

        # Default to only k=10 for NDCG@10
        self.k_values = [10]
        self.top_k = max(self.k_values)
        self.ignore_identical_ids = ignore_identical_ids
        self.main_metric = "NDCG@10"  # Primary metric for retrieval tasks
        self.corpus_chunk_size = corpus_chunk_size

    @classmethod
    def get_main_score_name(cls) -> str:
        """
        Get the name of the main score for retrieval tasks.

        Returns:
            The name of the main score metric.
        """
        return "NDCG@10"

    def __call__(
        self,
        model: Any,
        *,
        encode_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate retrieval performance with nDCG@10 as the primary metric.

        Args:
            model: The embedding/retrieval model to evaluate
            encode_kwargs: Additional encoding parameters
            **kwargs: Additional evaluation parameters including:
                - corpus: Dictionary mapping doc_id to document dict with 'title' and 'text'
                - queries: Dictionary mapping query_id to query text
                - qrels: Query relevance judgments mapping query_id -> doc_id -> relevance
                - score_function: Similarity function to use (default: "cosine")
                - task_name: Task name for instruction-tuned models
                - dataset_name: Dataset name for dataset-specific prompts

        Returns:
            Dictionary containing evaluation metrics with nDCG@10 as primary
        """
        # Extract required parameters from kwargs
        corpus = kwargs.get("corpus")
        queries = kwargs.get("queries")
        qrels = kwargs.get("qrels")
        score_function = kwargs.get("score_function", "cosine")
        task_name = kwargs.get("task_name")
        dataset_name = kwargs.get("dataset_name")

        if not all([corpus, queries, qrels]):
            raise ValueError(
                "corpus, queries, and qrels are required for retrieval evaluation"
            )
        assert corpus is not None
        assert queries is not None
        assert qrels is not None

        self._validate_inputs(corpus, queries, qrels)

        encode_kwargs = encode_kwargs or {}

        # Lock to cosine similarity for MTEB compliance (unless model has custom similarity)
        if not hasattr(model, "similarity"):
            if score_function != "cosine":
                print(
                    f"Warning: MTEB uses cosine similarity for dense retrieval. Overriding '{score_function}' -> 'cosine'"
                )
            score_function = "cosine"

        print(
            f"Evaluating retrieval on {len(queries)} queries and {len(corpus)} documents"
        )
        print(f"Primary metric: {self.main_metric}")
        print(f"Using top_k={self.top_k} for retrieval evaluation")

        # Ensure model is on GPU before evaluation
        self._ensure_gpu(model)

        # Check if this is a cross-encoder model
        if hasattr(model, "is_cross_encoder") and model.is_cross_encoder:
            raise NotImplementedError(
                "Cross-encoder evaluation is not implemented in this version."
            )
        else:
            # Dense retrieval model
            results = self._evaluate_dense_retrieval(
                model=model,
                corpus=corpus,
                queries=queries,
                score_function=score_function,
                encode_kwargs=encode_kwargs,
                task_name=task_name,
                dataset_name=dataset_name,
            )

        # Compute evaluation metrics
        metrics = self.evaluate_metrics(
            qrels, results, self.k_values, self.ignore_identical_ids
        )

        # Highlight the main metric
        if self.main_metric in metrics:
            print(f"  {self.main_metric}: {metrics[self.main_metric]:.4f}")

        return metrics

    def _validate_inputs(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        qrels: dict[str, dict[str, int]],
    ) -> None:
        """
        Validate retrieval inputs.

        Args:
            corpus: Document corpus mapping doc_id -> {title, text}
            queries: Query dictionary mapping query_id -> query_text
            qrels: Relevance judgments mapping query_id -> doc_id -> relevance

        Raises:
            ValueError: If inputs are invalid
        """
        if len(corpus) == 0:
            raise ValueError("Corpus cannot be empty")

        if len(queries) == 0:
            raise ValueError("Queries cannot be empty")

        if len(qrels) == 0:
            raise ValueError("Qrels cannot be empty")

        # Check that documents have content
        for doc_id, doc in corpus.items():
            if not doc.get("text") and not doc.get("title"):
                raise ValueError(f"Document '{doc_id}' must have 'text' or 'title'")

    def _evaluate_dense_retrieval(
        self,
        model: Any,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        score_function: str,
        encode_kwargs: dict[str, Any],
        task_name: str | None = None,
        dataset_name: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate using dense retrieval (bi-encoder) model.

        Args:
            model: Dense retrieval model
            corpus: Document corpus
            queries: Query dictionary
            score_function: Similarity function
            encode_kwargs: Encoding parameters
            task_name: Task name for instruction-tuned models
            dataset_name: Dataset name for instruction-tuned models

        Returns:
            Results dictionary mapping query_id -> doc_id -> score
        """
        print("Using dense retrieval evaluation")

        # Prepare corpus texts
        corpus_texts = []
        corpus_ids = []
        for doc_id, doc in corpus.items():
            # Combine title and text if both exist
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()

            if title and text:
                full_text = f"{title} {text}"
            else:
                full_text = title or text

            corpus_texts.append(full_text)
            corpus_ids.append(doc_id)

        # Prepare query texts
        query_texts = []
        query_ids = []
        for query_id, query_text in queries.items():
            query_texts.append(query_text)
            query_ids.append(query_id)

        # Check if we should use chunked encoding
        use_chunking = (
            self.corpus_chunk_size and len(corpus_texts) > self.corpus_chunk_size
        )

        corpus_prompt_type = PromptType.document
        if use_chunking:
            print(
                f"Using chunked corpus encoding with chunk_size={self.corpus_chunk_size}"
            )

            corpus_embeddings = self._encode_corpus_chunked(
                corpus_texts,
                model,
                encode_kwargs=encode_kwargs,
                prompt_type=corpus_prompt_type,
                task_name=task_name,
                dataset_name=dataset_name,
            )
        else:
            print(f"Encoding {len(corpus_texts)} documents...")
            corpus_embeddings = self._encode_sentences_retrieval(
                corpus_texts,
                model,
                step_info="Corpus",
                encode_kwargs=encode_kwargs,
                prompt_type=corpus_prompt_type,
                task_name=task_name or "",
                dataset_name=dataset_name or "",
            )

        print(f"Encoding {len(query_texts)} queries...")
        query_embeddings = self._encode_sentences_retrieval(
            query_texts,
            model,
            step_info="Queries",
            encode_kwargs=encode_kwargs,
            prompt_type=PromptType.query,
            task_name=task_name or "",
            dataset_name=dataset_name or "",
        )

        # Compute similarities and create results
        print("Computing similarities and ranking documents...")
        results: dict[str, dict[str, float]] = {}

        for i, query_id in enumerate(query_ids):
            query_emb = query_embeddings[i : i + 1]  # Keep 2D shape

            # Compute similarity with all documents
            similarities = self._compute_retrieval_similarities(
                query_emb[0], corpus_embeddings, score_function
            )

            # Sort and get top-k
            doc_scores = list(zip(corpus_ids, similarities))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Store top-k results
            results[query_id] = {}
            for doc_id, score in doc_scores[: self.top_k]:
                results[query_id][doc_id] = float(score)

        return results

    def _compute_retrieval_similarities(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        score_function: str,
    ) -> np.ndarray:
        """
        Compute similarities between one query and all corpus documents.

        Args:
            query_embedding: Query embedding vector (1D array)
            corpus_embeddings: Corpus embeddings matrix (N x D)
            score_function: Similarity function to use

        Returns:
            Array of similarity scores for each document
        """
        # Normalize score function to handle aliases
        score_function_aliases = {
            "cos_sim": "cosine",
            "cosine": "cosine",
            "cos": "cosine",
        }
        score_function = score_function_aliases.get(score_function, score_function)

        if score_function == "cosine":
            # Try to use sentence-transformers utilities first (matches MTEB approach)
            try:
                import torch
                from sentence_transformers.util import cos_sim

                # Convert to tensors if needed and ensure proper shape
                if not isinstance(query_embedding, torch.Tensor):
                    query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
                else:
                    query_tensor = query_embedding

                # Ensure query tensor is 2D (1, embedding_dim)
                if query_tensor.dim() == 1:
                    query_tensor = query_tensor.unsqueeze(0)

                if not isinstance(corpus_embeddings, torch.Tensor):
                    corpus_tensor = torch.tensor(corpus_embeddings, dtype=torch.float32)
                else:
                    corpus_tensor = corpus_embeddings

                # Compute cosine similarity using sentence-transformers utility
                # cos_sim returns (1, N) tensor, so we squeeze(0) to get (N,) array
                similarities = (
                    cos_sim(query_tensor, corpus_tensor).squeeze(0).cpu().numpy()
                )

            except ImportError:
                # Fall back to sklearn approach
                from sklearn.preprocessing import normalize

                # Normalize embeddings to unit length
                query_norm = normalize(query_embedding.reshape(1, -1), norm="l2")[0]
                corpus_norm = normalize(corpus_embeddings, norm="l2")

                # Compute dot product (cosine similarity)
                similarities = np.dot(corpus_norm, query_norm)

        elif score_function == "dot":
            # Try to use sentence-transformers utilities first
            try:
                import torch
                from sentence_transformers.util import dot_score

                # Convert to tensors if needed
                if not isinstance(query_embedding, torch.Tensor):
                    query_tensor = torch.tensor(query_embedding)
                else:
                    query_tensor = query_embedding

                if not isinstance(corpus_embeddings, torch.Tensor):
                    corpus_tensor = torch.tensor(corpus_embeddings)
                else:
                    corpus_tensor = corpus_embeddings

                # Compute dot product using sentence-transformers utility
                similarities = dot_score(query_tensor, corpus_tensor)[0].numpy()

            except ImportError:
                # Fall back to numpy approach
                similarities = np.dot(corpus_embeddings, query_embedding)

        elif score_function == "euclidean":
            # Try to use sentence-transformers utilities first
            try:
                import torch
                from sentence_transformers.util import euclidean_sim

                # Convert to tensors if needed
                if not isinstance(query_embedding, torch.Tensor):
                    query_tensor = torch.tensor(query_embedding)
                else:
                    query_tensor = query_embedding

                if not isinstance(corpus_embeddings, torch.Tensor):
                    corpus_tensor = torch.tensor(corpus_embeddings)
                else:
                    corpus_tensor = corpus_embeddings

                # Compute euclidean similarity using sentence-transformers utility
                similarities = euclidean_sim(query_tensor, corpus_tensor)[0].numpy()

            except ImportError:
                # Fall back to manual euclidean distance
                distances = np.linalg.norm(corpus_embeddings - query_embedding, axis=1)
                similarities = -distances

        elif score_function == "manhattan":
            # Try to use sentence-transformers utilities first
            try:
                import torch
                from sentence_transformers.util import manhattan_sim

                # Convert to tensors if needed
                if not isinstance(query_embedding, torch.Tensor):
                    query_tensor = torch.tensor(query_embedding)
                else:
                    query_tensor = query_embedding

                if not isinstance(corpus_embeddings, torch.Tensor):
                    corpus_tensor = torch.tensor(corpus_embeddings)
                else:
                    corpus_tensor = corpus_embeddings

                # Compute manhattan similarity using sentence-transformers utility
                similarities = manhattan_sim(query_tensor, corpus_tensor)[0].numpy()

            except ImportError:
                # Fall back to manual manhattan distance
                distances = np.sum(np.abs(corpus_embeddings - query_embedding), axis=1)
                similarities = -distances

        else:
            raise ValueError(f"Unsupported similarity metric: {score_function}")

        return similarities

    @staticmethod
    def evaluate_metrics(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        ignore_identical_ids: bool = False,
    ) -> dict[str, Any]:
        """
        Compute NDCG@10 evaluation metric only.

        Args:
            qrels: Query relevance judgments
            results: Retrieval results
            k_values: List of k values for evaluation (only k=10 supported)
            ignore_identical_ids: Whether to ignore identical query and document IDs

        Returns:
            Dictionary containing only NDCG@10 metric
        """
        if ignore_identical_ids:
            print("Ignoring identical query and document IDs for evaluation")
            # Remove identical ids from results
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
        else:
            print("Including identical query and document IDs in evaluation")

        # Initialize NDCG containers only
        all_ndcgs: dict[str, list[float]] = {f"NDCG@{k}": [] for k in k_values}

        # Prepare pytrec_eval evaluator for NDCG only
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string})
        scores = evaluator.evaluate(results)

        # Collect per-query NDCG scores
        for query_id in scores.keys():
            for k in k_values:
                all_ndcgs[f"NDCG@{k}"].append(scores[query_id][f"ndcg_cut_{k}"])

        # Compute mean NDCG metrics
        ndcg = {}
        for k in k_values:
            # Keep metrics in 0-1 range - percentage conversion handled centrally
            ndcg[f"NDCG@{k}"] = round((sum(all_ndcgs[f"NDCG@{k}"]) / len(scores)), 5)

        return ndcg

    def _encode_corpus_chunked(
        self,
        corpus_texts: list[str],
        model: Any,
        encode_kwargs: dict[str, Any],
        prompt_type: PromptType,
        task_name: str | None = None,
        dataset_name: str | None = None,
    ) -> np.ndarray:
        """
        Encode corpus in chunks to reduce memory usage while maintaining exact results.

        Args:
            corpus_texts: List of corpus texts to encode
            model: Embedding model
            encode_kwargs: Encoding parameters
            task_name: Task name for instruction-tuned models

        Returns:
            Concatenated embeddings array (same as non-chunked version)
        """

        print(
            f"Encoding {len(corpus_texts)} documents in chunks of {self.corpus_chunk_size}..."
        )

        all_embeddings = []
        chunk_size = self.corpus_chunk_size or 1000  # Default chunk size fallback
        num_chunks = (len(corpus_texts) + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(corpus_texts))
            chunk_texts = corpus_texts[chunk_start:chunk_end]

            step_info = (
                f"Chunk {chunk_idx + 1}/{num_chunks} ({chunk_start + 1}-{chunk_end})"
            )
            print(f"  {step_info}: Encoding {len(chunk_texts)} documents...")

            # Encode this chunk
            chunk_embeddings = self._encode_sentences_retrieval(
                chunk_texts,
                model,
                step_info=step_info,
                encode_kwargs=encode_kwargs,
                prompt_type=prompt_type,
                task_name=task_name or "",
                dataset_name=dataset_name or "",
            )

            all_embeddings.append(chunk_embeddings)

        # Concatenate all embeddings (this gives identical results to non-chunked)
        corpus_embeddings = np.vstack(all_embeddings)
        print(
            f"✓ Chunked encoding complete: {corpus_embeddings.shape[0]} documents -> {corpus_embeddings.shape[1]}D embeddings"
        )

        return corpus_embeddings

    def _encode_sentences_retrieval(
        self,
        sentences: list[str],
        embedding_model: Any,
        emb_model_name: str | None = None,
        step_info: str = "",
        encode_kwargs: dict[str, Any] | None = None,
        prompt_type: Any = None,
        task_name: str | None = None,
        dataset_name: str | None = None,
    ) -> np.ndarray:
        """
        Encode sentences for retrieval tasks with prompt type handling.

        For retrieval tasks, we need different prompts for queries vs corpus documents.
        This method implements the custom prompt logic:
        - Call get_emb_prompt() to check for custom prompts
        - If dict with "query"/"corpus" keys: use the appropriate prompt text
        - Otherwise: use PromptType.query or PromptType.document

        Args:
            sentences: List of sentences to encode
            embedding_model: The embedding model
            emb_model_name: Name of the model for logging purposes
            step_info: Optional progress information
            encode_kwargs: Optional keyword arguments to pass to the model's encode method
            prompt_type: PromptType.query or PromptType.document to distinguish encoding type
            task_name: Task name for get_emb_prompt
            dataset_name: Dataset name for get_emb_prompt

        Returns:
            NumPy array of sentence embeddings
        """
        from ..prompts import get_emb_prompt

        if not hasattr(embedding_model, "encode"):
            raise ValueError("Embedding model must have an 'encode' method")

        # Fallback if model_name not provided
        if emb_model_name is None:
            emb_model_name = getattr(embedding_model, "model_name", "unknown_model")

        step_prefix = f"[{step_info}] " if step_info else ""

        try:
            print(
                f"{step_prefix}Encoding {len(sentences)} sentences with {emb_model_name}..."
            )

            # Prepare encoding kwargs
            final_encode_kwargs = dict(encode_kwargs) if encode_kwargs else {}

            # Add batch_size if available
            if hasattr(embedding_model, "batch_size"):
                final_encode_kwargs["batch_size"] = embedding_model.batch_size

            # Add progress bar for SentenceTransformer
            final_encode_kwargs["show_progress_bar"] = True

            # Pass device="cuda" to ensure GPU usage
            final_encode_kwargs["device"] = "cuda"

            # Get prompt configuration from get_emb_prompt
            prompt_result = get_emb_prompt(
                emb_model=embedding_model,
                task_name=task_name or "retrieval",
                dataset_name=dataset_name or "",
            )

            # Determine which prompt to use based on retrieval-specific logic
            prompt_text = None
            prompt_name = None

            if isinstance(prompt_result, dict):
                # For retrieval tasks, check for specific keys
                if prompt_type == PromptType.query and "query" in prompt_result:
                    prompt_text = prompt_result["query"]
                elif prompt_type == PromptType.document and "corpus" in prompt_result:
                    prompt_text = prompt_result["corpus"]
                elif prompt_type == PromptType.document and "document" in prompt_result:
                    prompt_text = prompt_result["document"]
                else:
                    # Dict doesn't have the required key, use PromptType
                    prompt_name = prompt_type
            else:
                # Not a dict (could be string or None), use PromptType for retrieval
                prompt_name = prompt_type

            # Encode sentences with the determined prompt
            if prompt_text is not None:
                # Use custom prompt text
                embeddings = embedding_model.encode(
                    sentences, prompt=prompt_text, **final_encode_kwargs
                )
            elif prompt_name is not None:
                # Use PromptType for models that support it
                embeddings = embedding_model.encode(
                    sentences, prompt_name=prompt_name, **final_encode_kwargs
                )
            else:
                # No prompt
                embeddings = embedding_model.encode(sentences, **final_encode_kwargs)

            # Verify we got embeddings
            if embeddings is None:
                raise ValueError(
                    f"Encoding returned None for {len(sentences)} sentences"
                )

            embeddings_array = np.array(embeddings)

            # Verify embedding shape
            if len(embeddings_array.shape) != 2:
                raise ValueError(
                    f"Expected 2D embeddings, got shape {embeddings_array.shape}"
                )

            if embeddings_array.shape[0] != len(sentences):
                raise ValueError(
                    f"Expected {len(sentences)} embeddings, got {embeddings_array.shape[0]}"
                )

            print(
                f"{step_prefix}✓ Successfully encoded {embeddings_array.shape[0]} sentences to {embeddings_array.shape[1]}D embeddings"
            )

            return embeddings_array

        except (
            RuntimeError,
            ValueError,
            TypeError,
            MemoryError,
            OSError,
        ) as e:
            error_msg = f"{step_prefix}Failed to encode {len(sentences)} sentences with {emb_model_name}: {str(e)}"
            print(f"ERROR: {error_msg}")

            raise ValueError(error_msg) from e

        finally:
            pass
