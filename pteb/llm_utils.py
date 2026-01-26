import re
import time
from typing import Any

import ollama


def get_pre_prompt(
    api_name: str,
    gen_model_name: str,
    custom_prompt: str | None = None,
    languages: str | list[str] | None = None,
    dataset_name: str | None = None,
) -> str:
    """
    Returns a suitable pre-prompt based on the generative model name.
    If the model name is not recognized, prompts the user to input one.

    Args:
        api_name (str): The name of the API (e.g., "Ollama").
        gen_model_name (str): The name of the generative model (e.g., "gpt", "llama").
        custom_prompt (str, optional): Custom paraphrase prompt to use instead of default.
                                      If provided, this overrides model-specific prompts.
        languages (str or list, optional): Language code(s) of the text to be paraphrased.
                                          Can be a string like "en" or a list like ["EN", "DE"].
                                          If None and dataset_name is provided, will check dataset metadata.
                                          If non-English, adds language preservation instruction.
        dataset_name (str, optional): Name of the dataset being processed. Used to check
                                     non-English status from metadata when languages is None.

    Returns:
        str: The corresponding pre-prompt.
    """
    # If custom prompt is provided, use it instead of model-specific defaults
    if custom_prompt is not None:
        return custom_prompt
    gen_model_name = gen_model_name.lower()
    if "gpt" in gen_model_name:
        pre_prompt = (
            "Rephrase the following text while maintaining its original meaning. "
            "If the text contains only a single word, provide a definition or a synonym. "
            "When done, check and make sure that the length of the original is approximately maintained. "
        )
    elif "llama" in gen_model_name:
        pre_prompt = (
            "Rephrase the following text while maintaining its original meaning. "
            "When done, make sure to only return the paraphrased text (e.g. no explanations) and "
            "to only provide a single paraphrase (i.e., no alternatives). Do not include any other text in your response like notes or explanations. Do not start with 'Paraphrase:', 'Paraphrase:' or 'Here's the rephrased text:'. Do not break down the changes in the text. "
        )
    else:
        pre_prompt = "Rephrase the following text while keeping its original meaning. Only reply with the paraphrased text and only provide a single response - no alternatives! Do not include thinking tokens, explanations or notes. "

    # Determine if we should add multilingual prompt extension
    should_add_multilingual_extension = False

    # Add language preservation instruction for non-English languages
    # Handle both string and list inputs
    if languages:
        # Convert to list if it's a string
        lang_list = [languages] if isinstance(languages, str) else languages
        # Check if any non-English language is present
        # Convert all to lowercase for comparison
        lang_lower = [lang.lower() for lang in lang_list]
        is_english_only = all(
            lang in ["en", "english", "en-en", "en-ext"] for lang in lang_lower
        )
        if not is_english_only:
            should_add_multilingual_extension = True
    elif dataset_name:
        # No languages specified, check dataset metadata
        try:
            from .dataset_metadata import get_dataset_metadata

            metadata = get_dataset_metadata(dataset_name)
            # Only add multilingual extension if dataset contains non-English data
            if metadata.is_non_en():
                should_add_multilingual_extension = True
        except (ValueError, ImportError):
            # If metadata not found or import fails, don't add multilingual extension
            pass

    if should_add_multilingual_extension:
        pre_prompt += "Answer in the same language as the input text. Do not translate to English or any other language. "

    pre_prompt += "Text:"
    return pre_prompt


def ask_llm(
    text: str,
    pre_prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    seed: int | None = None,
    structured_output_spec: dict[str, Any] | None = None,
) -> str:
    """
    A single interface for making a single request to Ollama LLM service.

    Args:
        text (str): The text to be processed/paraphrased.
        pre_prompt (str): The instruction/prompt prefix as system message.
        model (str): The model name (e.g. "llama-2").
        temperature (float): Temperature for sampling.
        top_p (float): Nucleus sampling parameter.
        seed (int): Random seed for reproducibility.
        structured_output_spec (dict): A dictionary specifying structured output formatting,
                                  e.g., {"format": {"type": "json_schema", ...}}.

    Returns:
        str: The text response from the model.
    """

    response = ask_ollama(
        text=text,
        pre_prompt=pre_prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        use_structured_output=structured_output_spec is not None,
    )

    return response


def ask_llm_with_retry(
    text: str,
    pre_prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    seed: int | None = None,
    structured_output_spec: dict[str, Any] | None = None,
    verbose: int = 0,
    wait_time: int = 60,
) -> str:
    """
    Retries an Ollama API call with a delay if it fails.

    Args:
        text (str): The text to be processed/paraphrased.
        pre_prompt (str): The instruction/prompt prefix as system message.
        model (str): The model name.
        temperature (float): Temperature for sampling.
        top_p (float): Nucleus sampling parameter.
        seed (int): Random seed for reproducibility.
        structured_output_spec (dict): A dictionary specifying structured output formatting.
        verbose (int): Verbosity level for logging retries (0 = silent, 1 = show messages).
        wait_time (int): Seconds to wait between retries.

    Returns:
        str: The text response from the model.
    """
    max_attempts = 5
    n_attempts = 0
    while True:
        try:
            response = ask_llm(
                text=text,
                pre_prompt=pre_prompt,
                model=model,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                structured_output_spec=structured_output_spec,
            )
            # Remove all text between thinking tokens "<think>" and "</think>" incl. newlines
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            # Check if response is too long (is the case when Qwen3 inlcude thinking tokens in its reply)
            if len(response) <= 5 * len(text):
                break
            else:
                if verbose >= 1:
                    print("Warning: Response is too long. Retrying...")
                n_attempts += 1
                if n_attempts > max_attempts:
                    if verbose >= 1:
                        print(
                            "WARNING: Max attempts reached. Response is still too long.\n Original text: "
                            + text
                            + "\n Response: "
                            + response
                            + "\n - End of response. Returning original text."
                        )
                    response = text
                    break
        except Exception as e:
            # Check for Ollama connection error (should not retry)
            if "Failed to connect to Ollama" in str(e):
                if verbose >= 1:
                    print(f"Ollama connection failed: {e}")
                raise ConnectionError(f"Failed to connect to Ollama: {e}") from e

            # Check for Ollama model not found error (should not retry)
            if hasattr(e, "status_code") and e.status_code == 404:
                if verbose >= 1:
                    print(f"Ollama model not found: {e}")
                raise e  # Re-raise the error to stop execution

            n_attempts += 1
            if n_attempts > max_attempts:
                if verbose >= 1:
                    print(
                        f"Max attempts reached for error: {e}. Returning original text."
                    )
                return text
            if verbose == 1:
                print(
                    f"Error during Ollama API call: {e}. Retrying in {wait_time}s... (attempt {n_attempts}/{max_attempts})"
                )
            time.sleep(wait_time)

    return response.strip()


def ask_ollama(
    text: str,
    pre_prompt: str | None,
    model: str,
    temperature: float,
    top_p: float,
    seed: int | None = None,
    use_structured_output: bool = False,
) -> str:
    """Prompts the given Ollama model with the given pre_prompt and texts (either a single text or multiple) and returns the content of the result."""

    if pre_prompt is not None:
        messages = [
            {
                "role": "system",
                "content": pre_prompt,
            },
            {
                "role": "user",
                "content": text,
            },
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": text,
            },
        ]
    hyperparameters = {
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "think": False,
    }

    # Build chat kwargs
    chat_kwargs = {
        "model": model,
        "messages": messages,
        "options": hyperparameters,
    }

    # Add format="json" if structured output is enabled
    if use_structured_output:
        chat_kwargs["format"] = "json"

    response = ollama.chat(**chat_kwargs)

    response = response["message"]["content"]
    return response
