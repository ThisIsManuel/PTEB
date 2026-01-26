"""
Paraphrase-enhanced task base class for PTEB.

This module defines the ParaphraseTask class that extends BaseTask to include
paraphrase generation and evaluation capabilities.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from ..llm_utils import ask_llm_with_retry, get_pre_prompt
from .base_task import BaseTask


class ParaphraseTask(BaseTask):
    """
    Base class for tasks that involve paraphrase generation and evaluation.

    This class extends BaseTask to include paraphrase-specific functionality
    while maintaining the core task interface.
    """

    def __init__(
        self,
        task_name: str,
        task_description: str,
        datasets: List[str],
        config: Dict[str, Any],
        **kwargs,
    ):
        """
        Initialize the paraphrase task.

        Args:
            task_name: Name of the task
            task_description: Description of the task
            datasets: List of supported datasets
            config: Configuration dictionary
            **kwargs: Additional parameters
        """
        super().__init__(task_name, task_description, datasets, config, **kwargs)

        # Paraphrase-specific configuration
        self.save_paraphrases = config.get("save_paraphrases", False)
        self.n_runs = config.get("n_runs", 1)

        # Task type for directory structure
        self.task_type = self._get_task_type(task_name)

        # Generative model configuration - single model and API only (strings)
        self.gen_model = config.get("gen_model")  # String or None
        self.gen_model_api = "ollama"

        self.temperature = config.get("temperature", 0.8)
        self.top_p = config.get("top_p", 0.9)
        # Use paraphrase_prompt if available, fall back to pre_prompt for backward compatibility
        self.pre_prompt = config.get(
            "paraphrase_prompt", config.get("pre_prompt", None)
        )

        # Output configuration (hardcoded to JSONL)
        self.output_format_paraphrases = "jsonl"
        self.output_format_results = "jsonl"

    def _get_task_type(self, task_name: str) -> str:
        """
        Determine the task type directory name from the task name.

        Args:
            task_name: Name of the task

        Returns:
            Directory name for the task type
        """
        task_name_lower = task_name.lower()

        if "sts" in task_name_lower or "semantic" in task_name_lower:
            return "sts"
        elif "pair" in task_name_lower:
            return "pair_classification"
        elif "retrieval" in task_name_lower:
            return "retrieval"
        elif "classification" in task_name_lower and "pair" not in task_name_lower:
            return "classification"
        else:
            # Default fallback - use task name as-is
            return task_name_lower

    @abstractmethod
    def downsample_data(self, *args, **kwargs):
        """
        Downsample data according to task-specific requirements.

        Each task implements with its own signature based on data structures.
        Must check self.config.get("downsample") and return data unchanged if None.
        """
        pass

    def paraphrase_data(
        self,
        sentences1: list[str] | None = None,
        sentences2: list[str] | None = None,
        gen_model_name: str | None = None,
        gen_api_name: str | None = None,
        dataset_name: str | None = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate paraphrases for sentence pairs using the specified generative model.

        Args:
            sentences1: List of first sentences
            sentences2: List of second sentences
            gen_model_name: Name of the generative model
            gen_api_name: API name for the generative model
            dataset_name: Name of the dataset being processed

        Returns:
            Tuple of (paraphrased_sentences1, paraphrased_sentences2)
        """
        return self._generate_online_paraphrases(
            sentences1,
            sentences2,
            gen_model_name,
            gen_api_name,
            dataset_name,
        )

    def _generate_paraphrases(
        self,
        texts: List[str],
        gen_model_name: str,
        gen_api_name: str,
        dataset_name: str | None = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Generate paraphrases for a list of texts using a single seed.

        This method generates paraphrases for ONE run. For multiple runs with different
        seeds, call this method multiple times in a loop.

        Args:
            texts: List of texts to paraphrase
            gen_model_name: Name of the generative model
            gen_api_name: API name for the generative model
            dataset_name: Name of the dataset being processed (currently unused)
            seed: Seed for paraphrase generation. If None, uses self.seed_paraphrasing

        Returns:
            List of paraphrased texts (same order and length as input)
        """
        # Use provided seed or fall back to instance seed
        effective_seed = seed if seed is not None else self.seed_paraphrasing

        # Get pre-prompt (Ollama is always used)
        pre_prompt = get_pre_prompt(
            "ollama",
            gen_model_name,
            custom_prompt=self.pre_prompt,
            languages=self.config.get("languages", None),
            dataset_name=dataset_name,
        )

        paraphrased_texts = []

        # Generate paraphrases with progress bar
        for text in tqdm(
            texts, desc=f"Paraphrasing with {gen_model_name}", leave=False
        ):
            if text:
                # Generate paraphrase (using Ollama)
                response = ask_llm_with_retry(
                    pre_prompt=pre_prompt,
                    text=text,
                    model=gen_model_name,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=effective_seed,
                    wait_time=60,
                )
            else:
                response = ""

            paraphrased_texts.append(response.strip())

        return paraphrased_texts

    def _generate_online_paraphrases(
        self,
        sentences1: List[str],
        sentences2: List[str],
        gen_model_name: str,
        gen_api_name: str,
        dataset_name: str = "",
    ) -> Tuple[List[str], List[str]]:
        """
        Generate paraphrases online using the specified model.

        This method is now a wrapper around the new _generate_paraphrases method
        for backward compatibility. New code should use _generate_paraphrases directly.

        Args:
            sentences1: List of first sentences
            sentences2: List of second sentences
            gen_model_name: Name of the generative model
            gen_api_name: API name for the generative model
            dataset_name: Name of the dataset being processed (currently unused)

        Returns:
            Tuple of (paraphrased_sentences1, paraphrased_sentences2)
        """
        # Use the new method for both sentence lists
        paraphrases1 = self._generate_paraphrases(
            texts=sentences1,
            gen_model_name=gen_model_name,
            gen_api_name=gen_api_name,
            dataset_name=dataset_name,
        )

        paraphrases2 = self._generate_paraphrases(
            texts=sentences2,
            gen_model_name=gen_model_name,
            gen_api_name=gen_api_name,
            dataset_name=dataset_name,
        )

        return paraphrases1, paraphrases2

    @abstractmethod
    def evaluate(
        self,
        dataset_name: str,
        embedding_model: SentenceTransformer,
        emb_model_name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate the embedding model on paraphrased data.

        Args:
            dataset_name: Name of the dataset to evaluate on
            embedding_model: The embedding model to evaluate
            emb_model_name: Name of the embedding model (optional)
            **kwargs: Additional evaluation parameters

        This method must be implemented by concrete task classes
        to define task-specific evaluation logic.
        """
        pass
