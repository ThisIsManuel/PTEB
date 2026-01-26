"""
Abstract base evaluator for PTEB.

This module defines the BaseEvaluator class following the MTEB reference
implementation pattern with flexible __call__ interface.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from sentence_transformers.util import cos_sim
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Import memory utilities
from ..memory_utils import (
    calculate_optimal_batch_size,
    clear_gpu_memory,
    memory_efficient_context,
)
from ..prompts import get_emb_prompt

try:
    from mteb.abstasks.AbsTaskRetrieval import (
        PromptType,  # type: ignore[unresolved-import]
    )
except ImportError:
    # Define a fallback PromptType if mteb is not available
    class PromptType:
        query = "query"
        document = "document"


class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluation metric computations.

    Following MTEB's evaluator pattern with flexible __call__ interface
    that allows each evaluator to define its own input format.
    """

    def __init__(self, seed: int = 1337, **kwargs):
        """
        Initialize the base evaluator.

        Args:
            seed: Random seed for reproducibility (should be seed_general)
            **kwargs: Additional evaluator-specific parameters
        """
        # Use seed_general for all ML operations in evaluators
        self.seed: int = seed
        self.config: dict[str, Any] = kwargs

        # Set random seeds for reproducibility (following MTEB pattern)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if torch is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        # Only print if verbose is enabled
        if kwargs.get("verbose", True):
            print(f"  {self.__class__.__name__} initialized with seed {self.seed}")

    def _ensure_gpu(self, model, device="cuda:0"):
        """
        Ensure the model is on GPU before evaluation.

        Args:
            model: The model to check/move to GPU
            device: The target device (default: "cuda:0")
        """
        if hasattr(model, "to"):
            model.to(device)
        try:
            model.eval()
        except Exception:
            pass

        # Verify model is on GPU
        if hasattr(model, "device"):
            print(f"Model device: {model.device}")
        elif hasattr(model, "parameters"):
            try:
                device_check = next(model.parameters()).device
                print(f"Model device: {device_check}")
            except StopIteration:
                pass

    @abstractmethod
    def __call__(
        self, model: Any, *, encode_kwargs: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Evaluate the embedding model.

        This is called during evaluation to compute metrics. Each evaluator
        can define its own specific input format and parameters.

        Args:
            model: The embedding model to evaluate
            encode_kwargs: Keyword arguments to pass to the model's encode method
            **kwargs: Additional evaluation parameters specific to each evaluator

        Returns:
            Dictionary containing evaluation metrics
        """
        pass

    def _validate_inputs(self, *args, **kwargs) -> None:
        """
        Validate task-specific inputs.

        Each evaluator must override this to validate its specific input format.

        Raises:
            ValueError: If inputs are invalid
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _validate_inputs()"
        )

    def _encode_sentences(
        self,
        sentences: list[str],
        embedding_model: Any,
        emb_model_name: str | None = None,
        step_info: str = "",
        encode_kwargs: dict[str, Any] | None = None,
        prompt_text: str | dict[str, str] | None = None,
        task_name: str | None = None,
        dataset_name: str | None = None,
    ) -> np.ndarray | None:
        """
        Encode sentences using the provided embedding model with chunked processing for memory efficiency.

        Args:
            sentences: List of sentences to encode
            embedding_model: The embedding model
            emb_model_name: Name of the model for logging purposes
            step_info: Optional progress information (e.g., "1/4", "2/4")
            encode_kwargs: Optional keyword arguments to pass to the model's encode method
            prompt_text: Optional prompt text override
            task_name: Optional task name for instruction-tuned models
            dataset_name: Optional dataset name for instruction-tuned models

        Returns:
            NumPy array of sentence embeddings

        Raises:
            ValueError: If encoding fails or model doesn't have encode method
        """
        if not hasattr(embedding_model, "encode"):
            raise ValueError("Embedding model must have an 'encode' method")

        if not sentences:
            raise ValueError("Cannot encode empty sentence list")

        # Fallback if model_name not provided
        if emb_model_name is None:
            emb_model_name = getattr(embedding_model, "model_name", "unknown_model")

        step_prefix = f"[{step_info}] " if step_info else ""

        # Get chunking configuration from evaluator config
        chunk_size = self.config.get("encoding_chunk_size", None)

        # Determine if we need chunked processing
        if (not chunk_size) or (len(sentences) <= chunk_size):
            chunk_size = len(sentences)  # Process all at once
            print(f"Chunk size set to {chunk_size}, processing all sentences at once")
        else:
            print(
                f"{step_prefix}Using chunked encoding: {len(sentences)} sentences in chunks of {chunk_size}"
            )

        all_embeddings = []
        num_chunks = math.ceil(len(sentences) / chunk_size)

        # Create progress bar for chunks
        chunk_pbar = tqdm(
            total=num_chunks,
            desc=f"{step_prefix.strip()}Encoding chunks",
            unit="chunk",
            leave=False,
        )

        try:
            for i in range(0, len(sentences), chunk_size):
                chunk_idx = i // chunk_size + 1
                chunk_sentences = sentences[i : i + chunk_size]

                chunk_step_info = f"{step_prefix}Chunk {chunk_idx}/{num_chunks}"

                # Update progress bar description
                chunk_pbar.set_description(
                    f"{step_prefix.strip()}Encoding chunk {chunk_idx}/{num_chunks} ({len(chunk_sentences)} sentences)"
                )

                try:
                    # Process chunk with memory management
                    chunk_embeddings = self._encode_sentences_chunk(
                        chunk_sentences,
                        embedding_model,
                        emb_model_name,
                        chunk_step_info,
                        encode_kwargs,
                        prompt_text,
                        task_name,
                        dataset_name,
                    )

                    if chunk_embeddings is not None:
                        all_embeddings.append(chunk_embeddings)
                        chunk_pbar.update(1)  # Update progress bar
                    else:
                        # None indicates OOM - stop processing entire dataset
                        chunk_pbar.set_description(
                            f"{step_prefix.strip()}OOM error - aborting"
                        )
                        chunk_pbar.close()
                        clear_gpu_memory(aggressive=True, sync=True)
                        return None

                except Exception as e:
                    chunk_pbar.set_description(
                        f"{step_prefix.strip()}Error in chunk {chunk_idx}"
                    )
                    chunk_pbar.close()
                    # For non-OOM errors, also abort the entire dataset
                    clear_gpu_memory()
                    raise e

        finally:
            chunk_pbar.close()

        # Combine all embeddings
        if not all_embeddings:
            raise ValueError(
                f"All chunks failed to encode for {len(sentences)} sentences"
            )

        combined_embeddings = np.vstack(all_embeddings)

        print(
            f"{step_prefix}✓ Successfully encoded {combined_embeddings.shape[0]} sentences to {combined_embeddings.shape[1]}D embeddings using {len(all_embeddings)} chunks"
        )

        return combined_embeddings

    def _encode_sentences_chunk(
        self,
        sentences: list[str],
        embedding_model: Any,
        emb_model_name: str,
        step_prefix: str,
        encode_kwargs: dict[str, Any] | None = None,
        prompt_text: str | dict[str, str] | None = None,
        task_name: str | None = None,
        dataset_name: str | None = None,
    ) -> np.ndarray | None:
        """
        Encode a single chunk of sentences with memory management.

        Args:
            sentences: List of sentences to encode
            embedding_model: The embedding model
            emb_model_name: Name of the model for logging purposes
            step_prefix: Progress information prefix
            encode_kwargs: Optional keyword arguments to pass to the model's encode method
            prompt_text: Optional prompt text override
            task_name: Optional task name for instruction-tuned models
            dataset_name: Optional dataset name for instruction-tuned models

        Returns:
            NumPy array of sentence embeddings
        """
        try:
            # Prepare encoding kwargs with prompt_type and task_name if provided
            final_encode_kwargs = dict(encode_kwargs) if encode_kwargs else {}

            # Optimize batch size based on available memory
            original_batch_size = final_encode_kwargs.get("batch_size", 32)
            if hasattr(embedding_model, "batch_size"):
                original_batch_size = embedding_model.batch_size

            optimal_batch_size = calculate_optimal_batch_size(
                original_batch_size, memory_threshold_gb=1.5, min_batch_size=1
            )
            final_encode_kwargs["batch_size"] = optimal_batch_size

            if optimal_batch_size != original_batch_size:
                print(
                    f"{step_prefix} Adjusted batch size from {original_batch_size} to {optimal_batch_size} for memory efficiency"
                )

            # Disable progress bar for individual chunks to avoid clutter with main chunk progress bar
            final_encode_kwargs["show_progress_bar"] = False

            # Add normalize_embeddings parameter if configured
            if self.config.get("normalize_embeddings", False):
                final_encode_kwargs["normalize_embeddings"] = True

            # Get prompt if not provided
            if prompt_text is None:
                prompt_text = get_emb_prompt(
                    emb_model=embedding_model,
                    task_name=task_name,
                    dataset_name=dataset_name,
                )

            # Use memory-efficient context for encoding
            with memory_efficient_context(
                clear_before=True, clear_after=True, aggressive=True
            ):
                # Tasks like retrieval may have different prompts for query vs document
                if prompt_text:
                    embeddings = embedding_model.encode(
                        sentences, prompt=prompt_text, **final_encode_kwargs
                    )
                else:
                    embeddings = embedding_model.encode(
                        sentences, **final_encode_kwargs
                    )

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

            return embeddings_array

        except torch.OutOfMemoryError as e:
            error_msg = f"{step_prefix}CUDA OOM: Failed to encode {len(sentences)} sentences with {emb_model_name}: {str(e)}"
            print(f"ERROR: {error_msg}")

            # Aggressive memory cleanup on OOM
            clear_gpu_memory(aggressive=True, sync=True)

            print(f"{step_prefix}Skipping this chunk due to CUDA OOM - returning None")
            return None

        except (
            RuntimeError,
            ValueError,
            TypeError,
            MemoryError,
            OSError,
        ) as e:
            error_msg = f"{step_prefix}Failed to encode {len(sentences)} sentences with {emb_model_name}: {str(e)}"
            print(f"ERROR: {error_msg}")

            # Clear GPU memory on failure
            clear_gpu_memory()

            raise ValueError(error_msg) from e

        finally:
            # Ensure cleanup after encoding regardless of outcome
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _compute_similarities(
        self,
        embeddings1: torch.Tensor | np.ndarray,
        embeddings2: torch.Tensor | np.ndarray,
        similarity_metric: str = "cosine",
        embedding_model: Any = None,
        model_name: str | None = None,
    ) -> np.ndarray:
        """
        Compute pairwise similarities between embeddings.

        Following official MTEB approach: First try model-specific similarity method,
        then fall back to standard similarity metrics.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            similarity_metric: Type of similarity ("cosine", "dot", "euclidean", "manhattan")
            embedding_model: The embedding model (used for model-specific similarity if available)

        Returns:
            Array of similarity scores
        """
        # Fallback if model_name not provided
        if model_name is None:
            model_name = getattr(embedding_model, "model_name", "unknown_model")

        # First try model-specific similarity method (following official MTEB)
        if embedding_model and hasattr(embedding_model, "similarity_pairwise"):
            try:
                similarities = embedding_model.similarity_pairwise(
                    embeddings1, embeddings2
                )
                return np.array(similarities)
            except Exception as e:
                print(
                    f"Model-specific similarity failed ({e}), falling back to {similarity_metric} similarity"
                )

        # Fall back to standard similarity metrics
        if similarity_metric == "cosine":
            # Try to use sentence-transformers utilities first (matches MTEB approach)
            try:
                # Convert to tensors if needed
                if not isinstance(embeddings1, torch.Tensor):
                    embeddings1 = torch.tensor(embeddings1)
                if not isinstance(embeddings2, torch.Tensor):
                    embeddings2 = torch.tensor(embeddings2)

                # Compute cosine similarity using sentence-transformers utility
                similarities = cos_sim(embeddings1, embeddings2).diagonal()
                if isinstance(similarities, torch.Tensor):
                    similarities = similarities.cpu().float().numpy()
                else:
                    similarities = np.array(similarities)
                return similarities

            except ImportError:
                # Fall back to sklearn approach
                if cosine_similarity is None:
                    raise ImportError(
                        "sklearn is required for cosine similarity computation"
                    )

                # Optimized cosine similarity computation (more similar to official MTEB)
                # Normalize embeddings first for more stable computation
                from sklearn.preprocessing import normalize

                # Normalize embeddings to unit length
                emb1_norm = normalize(embeddings1, norm="l2")
                emb2_norm = normalize(embeddings2, norm="l2")

                # Compute dot product of normalized embeddings (equivalent to cosine similarity)
                similarities = np.sum(emb1_norm * emb2_norm, axis=1)

                return similarities

        elif similarity_metric == "dot":
            # Try to use sentence-transformers utilities first
            try:
                from sentence_transformers.util import pairwise_dot_score

                # Convert to tensors if needed
                if not isinstance(embeddings1, torch.Tensor):
                    embeddings1 = torch.tensor(embeddings1)
                if not isinstance(embeddings2, torch.Tensor):
                    embeddings2 = torch.tensor(embeddings2)

                # Use pairwise dot score for one-to-one comparisons
                similarities = pairwise_dot_score(embeddings1, embeddings2).numpy()
                return similarities

            except ImportError:
                # Fall back to manual dot product
                similarities = []
                for emb1, emb2 in zip(embeddings1, embeddings2):
                    sim = np.dot(emb1, emb2)
                    similarities.append(sim)
                return np.array(similarities)

        elif similarity_metric == "euclidean":
            # Try to use sentence-transformers utilities first
            try:
                from sentence_transformers.util import pairwise_euclidean_sim

                # Convert to tensors if needed
                if not isinstance(embeddings1, torch.Tensor):
                    embeddings1 = torch.tensor(embeddings1)
                if not isinstance(embeddings2, torch.Tensor):
                    embeddings2 = torch.tensor(embeddings2)

                # Use pairwise euclidean similarity (already returns negative distances)
                similarities = pairwise_euclidean_sim(embeddings1, embeddings2).numpy()
                return similarities

            except ImportError:
                # Fall back to manual euclidean distance
                similarities = []
                for emb1, emb2 in zip(embeddings1, embeddings2):
                    dist = np.linalg.norm(emb1 - emb2)
                    similarities.append(-dist)
                return np.array(similarities)

        elif similarity_metric == "manhattan":
            # Try to use sentence-transformers utilities first
            try:
                from sentence_transformers.util import pairwise_manhattan_sim

                # Convert to tensors if needed
                if not isinstance(embeddings1, torch.Tensor):
                    embeddings1 = torch.tensor(embeddings1)
                if not isinstance(embeddings2, torch.Tensor):
                    embeddings2 = torch.tensor(embeddings2)

                # Use pairwise manhattan similarity (already returns negative distances)
                similarities = pairwise_manhattan_sim(embeddings1, embeddings2).numpy()
                return similarities

            except ImportError:
                # Fall back to manual manhattan distance
                similarities = []
                for emb1, emb2 in zip(embeddings1, embeddings2):
                    dist = np.sum(np.abs(emb1 - emb2))
                    similarities.append(-dist)
                return np.array(similarities)

        else:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

    def __str__(self) -> str:
        """String representation of the evaluator."""
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        """Detailed string representation of the evaluator."""
        return f"{self.__class__.__name__}(config={self.config})"
