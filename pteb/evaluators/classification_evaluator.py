"""
Classification evaluator for PTEB.

Supports LogisticRegression evaluation following MTEB methodology.
"""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .base_evaluator import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for classification tasks using LogisticRegression.

    Follows MTEB methodology with LogisticRegression as the primary classifier.
    """

    def __init__(self, max_iter: int = 100, seed: int = 1337, **kwargs):
        """
        Initialize the classification evaluator.

        Args:
            max_iter: Maximum iterations for logistic regression
            seed: Random seed for reproducibility
            **kwargs: Additional configuration parameters
        """
        super().__init__(seed=seed, **kwargs)
        self.max_iter = max_iter

    @classmethod
    def get_main_score_name(cls) -> str:
        """
        Get the name of the main score for classification tasks.

        Returns:
            The name of the main score metric.
        """
        return "accuracy"

    def __call__(
        self,
        model: Any,
        *,
        train_sentences: list[str] | None = None,
        train_labels: list[int] | None = None,
        test_sentences: list[str] | None = None,
        test_labels: list[int] | None = None,
        encode_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Evaluate the embedding model on classification data.

        Args:
            model: The embedding model to evaluate
            train_sentences: List of training sentences
            train_labels: List of training labels
            test_sentences: List of test sentences
            test_labels: List of test labels

        Returns:
            Dictionary with evaluation metrics
        """
        # Validate required inputs
        if (
            train_sentences is None
            or train_labels is None
            or test_sentences is None
            or test_labels is None
        ):
            raise ValueError(
                "train_sentences, train_labels, test_sentences, and test_labels are required"
            )

        # Validate inputs
        self._validate_inputs(
            train_sentences, train_labels, test_sentences, test_labels
        )

        # Ensure model is on GPU before evaluation
        self._ensure_gpu(model)

        # Get model name for logging
        model_name = getattr(model, "model_name", "unknown_model")

        # Encode sentences using base evaluator's method with task-specific prompts
        X_train = self._encode_sentences(
            train_sentences,
            model,
            model_name,
            "Train 1/2",
            encode_kwargs or {},
            task_name="Classification",
        )
        X_test = self._encode_sentences(
            test_sentences,
            model,
            model_name,
            "Test 2/2",
            encode_kwargs or {},
            task_name="Classification",
        )

        # Check for OOM errors during encoding
        if X_train is None or X_test is None:
            print("ERROR: CUDA OOM occurred during encoding - returning error result")
            return {
                "error": "CUDA_OOM",
                "error_message": "CUDA out of memory during sentence encoding",
                "accuracy": 0.0,
                "f1": 0.0,
                "f1_macro": 0.0,
            }

        # Convert labels to numpy arrays
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        # Initialize logistic regression classifier (matching MTEB parameters)
        clf = LogisticRegression(
            random_state=self.seed,
            n_jobs=1,  # Force single-process training to avoid debugger/joblib spawn issues
            max_iter=self.max_iter,
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # Calculate main metric only in 0-1 range (percentage conversion handled centrally)
        scores = {}
        scores["accuracy"] = accuracy_score(y_test, y_pred)

        return scores

    def _validate_inputs(
        self,
        train_sentences: list[str],
        train_labels: list[int],
        test_sentences: list[str],
        test_labels: list[int],
    ) -> None:
        """
        Validate classification input data.

        Args:
            train_sentences: Training sentences
            train_labels: Training labels
            test_sentences: Test sentences
            test_labels: Test labels

        Raises:
            ValueError: If input validation fails
        """
        if len(train_sentences) != len(train_labels):
            raise ValueError(
                f"Training sentences and labels must have same length: "
                f"{len(train_sentences)} vs {len(train_labels)}"
            )

        if len(test_sentences) != len(test_labels):
            raise ValueError(
                f"Test sentences and labels must have same length: "
                f"{len(test_sentences)} vs {len(test_labels)}"
            )

        if len(train_sentences) == 0:
            raise ValueError("Training data cannot be empty")

        if len(test_sentences) == 0:
            raise ValueError("Test data cannot be empty")
