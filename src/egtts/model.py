"""Model loading and inference utilities for Text-to-SQL generation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float16
):
    """
    Load Qwen2.5-Coder model and tokenizer.

    The model is loaded in FP16 to fit in NVIDIA L4 24GB VRAM:
    - Model: ~14GB (7B params × 2 bytes)
    - Remaining: ~10GB for context and generation

    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ("cuda" or "cpu")
        dtype: Model dtype (default: float16 for efficiency)

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading {model_name} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Load directly onto GPU with proper device mapping
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cuda:0",  # Explicit GPU device
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        model = model.to(device)

    model.eval()  # Set to evaluation mode

    print(f"Model loaded successfully. Device: {model.device}")
    print(f"Model hf_device_map: {getattr(model, 'hf_device_map', 'N/A')}")
    return model, tokenizer


def create_sql_prompt(question: str, schema: str, tokenizer=None, few_shot_examples: list = None) -> str | list:
    """
    Create a prompt for SQL generation given a question and database schema.

    Uses Qwen2.5-Coder's chat template for optimal performance.

    Args:
        question: Natural language question
        schema: Database schema (CREATE TABLE statements)
        tokenizer: Tokenizer to apply chat template (optional)
        few_shot_examples: Optional list of (question, evidence, sql) tuples for few-shot prompting

    Returns:
        Formatted prompt string or messages list
    """
    # Build system message with instructions
    system_message = """You are an expert SQL query generator. Given a database schema and a natural language question, generate the correct SQL query to answer the question.

Rules:
- Return ONLY the SQL query, no explanations
- Use proper SQL syntax for SQLite
- Use table aliases when joining multiple tables"""

    messages = [{"role": "system", "content": system_message}]

    # Add few-shot examples if provided
    if few_shot_examples:
        for ex_question, ex_evidence, ex_sql in few_shot_examples:
            # User turn with example question
            if ex_evidence:
                user_content = f"Question: {ex_question}\nHint: {ex_evidence}"
            else:
                user_content = f"Question: {ex_question}"
            messages.append({"role": "user", "content": user_content})
            # Assistant turn with example SQL
            messages.append({"role": "assistant", "content": ex_sql})

    # Final user message with target question and schema
    user_message = f"""Database schema:
{schema}

Question: {question}

SQL query:"""

    messages.append({"role": "user", "content": user_message})

    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    return messages


def generate_sql(
    model,
    tokenizer,
    prompt: str | list,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    do_sample: bool = False
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
        prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the generated portion
    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
    sql_lines = []
    for line in lines:
        line = line.strip()
        # Stop at explanatory sentences
        if line and not line.startswith("--") and not line.lower().startswith(("this query", "the query", "note:")):
            sql_lines.append(line)
        elif sql_lines:  # Stop after first SQL block
            break

    sql = " ".join(sql_lines).strip()

    return sql
