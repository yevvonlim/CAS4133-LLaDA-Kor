from typing import Dict, List
import random
from loguru import logger

_SYSTEM_PROMPT = "You are a helpful assistant."

def preprocess(
    examples: Dict[str, List],
    tokenizer,
    max_seq_length: int
) -> Dict[str, List[List[int]]]:
    """
    Turn raw (conversations, output) pairs into chat-style `input_ids`.

    Parameters
    ----------
    examples : Dict[str, List]
        Must contain:
          • "conversations": List of List[Dict[str, str]] each being
            [{"role": "system|user|assistant", "content": ...}, ...]
          • "output"       : List of assistant responses (str)
    tokenizer :
        Must implement:
          • apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            returning List[int] of token ids
          • encode(text, add_special_tokens=False) -> List[int]
    max_seq_length : int
        Maximum total sequence length (prompt + answer).

    Returns
    -------
    dict with:
      • input_ids      : List of List[int]
      • prompt_lengths : List[int]   (length of the prompt part only)
    """
    all_input_ids: List[List[int]] = []
    prompt_lengths: List[int] = []

    for conv, answer in zip(examples["instruction"], examples["output"]):
        # Build the full message list: system prompt + user/assistant turns
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(conv)

        # Tokenize the prompt with generation placeholder
        prompt_tokens: List[int] = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # include assistant tag token
            tokenize=True,
        )
        prompt_len = len(prompt_tokens)
        slots = max_seq_length - prompt_len
        if slots <= 0:
            logger.warning(
                f"[SKIP] max_seq_length={max_seq_length} insufficient for prompt_len={prompt_len}"
            )
            continue

        # Encode the answer, truncating if needed
        answer_tokens: List[int] = tokenizer.encode(
            answer, add_special_tokens=False
        )[:slots]

        # Concatenate prompt + answer
        input_ids = prompt_tokens + answer_tokens
        all_input_ids.append(input_ids)
        prompt_lengths.append(prompt_len)

    return {"input_ids": all_input_ids, "prompt_lengths": prompt_lengths}