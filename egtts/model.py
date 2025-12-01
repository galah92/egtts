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


def load_model(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float16,
    quantization: str | None = None,
) -> tuple[PreTrainedModel, Tokenizer]:
    """
    Load Qwen2.5-Coder model and tokenizer with optional quantization.

    Quantization options:
    - None (default): FP16 - 7B uses ~14GB, 14B uses ~28GB
    - "8bit": 8-bit quantization - 7B uses ~7GB, 14B uses ~14GB
    - "4bit": 4-bit quantization - 7B uses ~3.5GB, 14B uses ~7GB

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ("cuda" or "cpu")
        dtype: Model dtype (default: float16 for efficiency, ignored if quantized)
        quantization: Quantization mode ("8bit", "4bit", or None)

    Returns:
        Tuple of (model, tokenizer)
    """
    quant_str = f" with {quantization} quantization" if quantization else ""
    print(f"Loading {model_name} on {device}{quant_str}...")

    tokenizer: Tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    # Prepare quantization config if requested
    quantization_config: BitsAndBytesConfig | None = None
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
            "device_map": "cuda:0",
            "trust_remote_code": True,
        }
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = dtype

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True
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


def generate_sql(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    prompt: str | list[ChatMessage],
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    do_sample: bool = False,
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

    Returns:
        Generated SQL query string
    """
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
