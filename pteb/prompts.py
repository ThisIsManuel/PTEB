"""
Simplified prompt registry for MTEB tasks.
"""

from typing import Any

# Map MTEB task names to embeddinggemma prompt names (case-insensitive matching)
# embeddinggemma uses capitalized keys like "Classification", "Clustering", "STS"
TASK_NAME_MAPPING = {
    "classification": "Classification",
    "multilabelclassification": "MultilabelClassification",
    "clustering": "Clustering",
    "sts": "STS",
    "pair_classification": "Classification",  # Use classification prompt for pair classification
    # Note: Retrieval uses special handling with query/document keys
}


def get_emb_prompt(
    emb_model: Any,
    task_name: str | None,
    dataset_name: str | None,
) -> str | dict[str, str]:
    """
    Get model-specific prompt for the given task.

    This function returns ONLY model-defined prompts. If the model doesn't have
    a prompt for the task, it returns an empty string (no prompt).

    Args:
        emb_model: The embedding model (SentenceTransformer or similar)
        task_name: Task name (e.g., "sts", "classification", "retrieval")
        dataset_name: Dataset name (not used, kept for compatibility)

    Returns:
        - For retrieval tasks: Dict[str, str] with "query" and "document" keys
        - For other tasks: str (model's prompt or empty string if not defined)

    Examples:
        - all-mpnet-base-v2 (has prompts={'query': '', 'document': ''}):
          → Returns "" for STS task
          → Returns {"query": "", "document": ""} for retrieval
        - embeddinggemma-300m (has prompts={'STS': '...', ...}):
          → Returns model's STS prompt for STS task
          → Returns {"query": "...", "document": "..."} for retrieval
    """
    # Only use model-specific prompts - no task/dataset fallbacks
    # If model doesn't define a prompt for the task, use empty string
    model_prompts = getattr(emb_model, "prompts", None)

    if model_prompts is None:
        # Model has no prompts attribute - use empty string (no prompt)
        return ""

    # For retrieval tasks, check for query/document prompts separately
    if task_name.lower() == "retrieval":
        if "query" in model_prompts and "document" in model_prompts:
            # Return dict with query and document prompts
            # Even if they're empty strings, return the dict structure
            return {
                "query": model_prompts.get("query", ""),
                "document": model_prompts.get("document", ""),
            }
        else:
            # No retrieval-specific prompts - use empty string
            return ""

    # For non-retrieval tasks, look for task-specific prompt in model prompts
    # First try exact match (lowercase)
    prompt = model_prompts.get(task_name, None)

    # If not found, try mapped task name (e.g., "classification" -> "Classification")
    if prompt is None and task_name.lower() in TASK_NAME_MAPPING:
        mapped_task_name = TASK_NAME_MAPPING[task_name.lower()]
        prompt = model_prompts.get(mapped_task_name, None)

    # If still not found, use empty string (model has no prompt for this task)
    return prompt if prompt is not None else ""
