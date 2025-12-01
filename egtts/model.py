"""Model loading and inference utilities for Text-to-SQL generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if TYPE_CHECKING:
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )

    Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

# Type alias for chat messages
ChatMessage = dict[str, str]

# Arctic-Text2SQL system prompt (from paper Appendix C)
ARCTIC_SYSTEM_PROMPT = """You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine: SQLite

Instructions:
- Output only information that is explicitly asked for
- If the question asks for specific column(s), ensure your query returns exactly those column(s)
- Return complete information without gaps or extras
- Think through query construction steps beforehand"""

# Models that require Arctic-style prompting
ARCTIC_STYLE_MODELS = {"arctic-text2sql", "snowflake/arctic-text2sql-r1-7b"}


def is_arctic_model(model_name: str) -> bool:
    """Check if the model requires Arctic-style prompting."""
    model_lower = model_name.lower()
    return any(arctic in model_lower for arctic in ARCTIC_STYLE_MODELS)


# Model presets for easy switching
MODEL_PRESETS: dict[str, str] = {
    # General instruction models
    "qwen2.5-coder-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen3-coder-30b": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "qwen3-coder-30b-fp8": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3-8b": "Qwen/Qwen3-8B",
    # Text-to-SQL fine-tuned models
    "arctic-text2sql": "Snowflake/Arctic-Text2SQL-R1-7B",  # 68.9% BIRD
    "omnisql-7b": "seeklhy/OmniSQL-7B",  # Strong on BIRD, SQLite-only
}


def load_model(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype | Literal["auto"] = torch.float16,
    quantization: str | None = None,
) -> tuple[PreTrainedModel, Tokenizer]:
    """
    Load Qwen model and tokenizer with optional quantization.

    Supported models (use preset name or full HuggingFace path):
    - qwen2.5-coder-7b: Qwen/Qwen2.5-Coder-7B-Instruct (baseline)
    - qwen3-coder-30b: Qwen/Qwen3-Coder-30B-A3B-Instruct (MoE, 3B active)
    - qwen3-coder-30b-fp8: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 (FP8 quantized)
    - qwen3-4b: Qwen/Qwen3-4B-Instruct-2507 (small but powerful)
    - qwen3-8b: Qwen/Qwen3-8B (general Qwen3)

    Quantization options:
    - None (default): FP16 or auto dtype
    - "8bit": 8-bit quantization via bitsandbytes
    - "4bit": 4-bit quantization via bitsandbytes

    Args:
        model_name: HuggingFace model identifier or preset name
        device: Device to load model on ("cuda" or "cpu")
        dtype: Model dtype (default: float16, use "auto" for Qwen3 models)
        quantization: Quantization mode ("8bit", "4bit", or None)

    Returns:
        Tuple of (model, tokenizer)
    """
    # Resolve preset name to full model path
    if model_name.lower() in MODEL_PRESETS:
        model_name = MODEL_PRESETS[model_name.lower()]

    # Detect if this is a Qwen3 model or FP8 model (use auto dtype)
    is_qwen3 = "qwen3" in model_name.lower()
    is_fp8 = "-fp8" in model_name.lower()
    use_auto_dtype = is_qwen3 or is_fp8 or dtype == "auto"

    quant_str = f" with {quantization} quantization" if quantization else ""
    dtype_str = "auto" if use_auto_dtype else str(dtype)
    print(f"Loading {model_name} on {device}{quant_str} (dtype={dtype_str})...")

    tokenizer: Tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    # Prepare quantization config if requested (not for FP8 models)
    quantization_config: BitsAndBytesConfig | None = None
    if quantization and not is_fp8:
        if quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

    # Load directly onto GPU with proper device mapping
    model: PreTrainedModel
    if device == "cuda":
        load_kwargs: dict[str, Any] = {
            "device_map": "auto" if is_qwen3 else "cuda:0",
            "trust_remote_code": True,
        }
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        elif use_auto_dtype:
            load_kwargs["torch_dtype"] = "auto"
        else:
            load_kwargs["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto" if use_auto_dtype else dtype,
            trust_remote_code=True,
        )
        model = model.to(device)

    model.eval()  # Set to evaluation mode

    print(f"Model loaded successfully. Device: {model.device}")
    print(f"Model hf_device_map: {getattr(model, 'hf_device_map', 'N/A')}")
    return model, tokenizer


def create_sql_prompt(
    question: str,
    schema: str,
    tokenizer: Tokenizer | None = None,
    evidence: str | None = None,
    few_shot_examples: list[tuple[str, str, str]] | None = None,
) -> str | list[ChatMessage]:
    """
    Create a prompt for SQL generation given a question and database schema.

    Uses Qwen2.5-Coder's chat template for optimal performance.

    Args:
        question: Natural language question
        schema: Database schema (CREATE TABLE statements)
        tokenizer: Tokenizer to apply chat template (optional)
        evidence: Optional hint/evidence for the query
        few_shot_examples: Optional list of (question, evidence, sql) tuples for few-shot prompting

    Returns:
        Formatted prompt string (if tokenizer provided) or messages list
    """
    # Build prompt with clear structure
    parts: list[str] = [f"Given the following database schema:\n\n{schema}"]

    if evidence:
        parts.append(f"\nHint: {evidence}")

    parts.append(f"\nGenerate a SQL query to answer this question:\n{question}")
    parts.append(
        "\nReturn only the SQL query, without any explanation or markdown formatting."
    )

    user_message = "\n".join(parts)

    messages: list[ChatMessage] = [{"role": "user", "content": user_message}]

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return str(prompt)

    return messages


def create_arctic_prompt(
    question: str,
    schema: str,
    tokenizer: Tokenizer | None = None,
    evidence: str | None = None,
) -> str | list[ChatMessage]:
    """
    Create a prompt for Arctic-Text2SQL model.

    Arctic was trained with a specific format including system prompt,
    and expects output in <think> and <answer> tags.

    Args:
        question: Natural language question
        schema: Database schema (CREATE TABLE statements)
        tokenizer: Tokenizer to apply chat template (optional)
        evidence: Optional hint/evidence for the query

    Returns:
        Formatted prompt string (if tokenizer provided) or messages list
    """
    # Build user message with schema and question
    user_parts: list[str] = [f"Schema:\n{schema}"]

    if evidence:
        user_parts.append(f"\nHint: {evidence}")

    user_parts.append(f"\nQuestion: {question}")

    user_message = "\n".join(user_parts)

    messages: list[ChatMessage] = [
        {"role": "system", "content": ARCTIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return str(prompt)

    return messages


def extract_sql_from_arctic_response(response: str) -> str:
    """
    Extract SQL from Arctic-Text2SQL response format.

    Arctic outputs in format:
    <think>...reasoning...</think>
    <answer>
    ```sql
    SELECT ...
    ```
    </answer>

    Args:
        response: Raw model response

    Returns:
        Extracted SQL query
    """
    import re

    # Try to extract from <answer> tags first
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        # Extract SQL from markdown code block within answer
        sql_match = re.search(r"```sql\s*(.*?)\s*```", answer_content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        # Try generic code block
        sql_match = re.search(r"```\s*(.*?)\s*```", answer_content, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        # Return answer content as-is if no code block
        return answer_content

    # Fallback: try to extract SQL from markdown code block anywhere
    sql_match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()

    # Last resort: return the response after any <think> section
    think_match = re.search(r"</think>\s*(.*)", response, re.DOTALL)
    if think_match:
        remaining = think_match.group(1).strip()
        # Try to extract SQL from remaining
        sql_match = re.search(r"```sql\s*(.*?)\s*```", remaining, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        return remaining

    return response.strip()


def generate_sql(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    prompt: str | list[ChatMessage],
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    do_sample: bool = False,
    is_arctic: bool = False,
) -> str:
    """
    Generate SQL query from prompt using the loaded model.

    Args:
        model: Loaded causal LM model
        tokenizer: Loaded tokenizer
        prompt: Formatted prompt string or messages list
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (low = more deterministic)
        do_sample: Whether to use sampling (False = greedy decoding)
        is_arctic: Whether to use Arctic-specific output parsing

    Returns:
        Generated SQL query string
    """
    # Arctic needs more tokens for <think> + <answer> format
    if is_arctic and max_new_tokens < 2048:
        max_new_tokens = 2048
    # If prompt is a list of messages, apply chat template
    if isinstance(prompt, list):
        prompt_str = str(
            tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        )
    else:
        prompt_str = prompt

    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    input_length: int = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(  # type: ignore[operator]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated portion
    generated_ids = outputs[0][input_length:]
    generated_text: str = tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    # Use Arctic-specific extraction if needed
    if is_arctic:
        return extract_sql_from_arctic_response(generated_text)

    # Clean up SQL: remove markdown formatting, explanations, etc.
    sql = generated_text

    # Remove markdown code blocks
    if "```sql" in sql.lower():
        sql = sql.split("```sql", 1)[1].split("```")[0].strip()
    elif "```" in sql:
        # Try to extract from generic code block
        parts = sql.split("```")
        if len(parts) >= 2:
            sql = parts[1].strip()

    # Remove explanatory text (common patterns)
    # Take only the first SQL statement
    lines = sql.split("\n")
    sql_lines: list[str] = []
    for line in lines:
        line = line.strip()
        # Stop at explanatory sentences
        if (
            line
            and not line.startswith("--")
            and not line.lower().startswith(("this query", "the query", "note:"))
        ):
            sql_lines.append(line)
        elif sql_lines:  # Stop after first SQL block
            break

    sql = " ".join(sql_lines).strip()

    return sql
