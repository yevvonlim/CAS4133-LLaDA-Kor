from loguru import logger


def preprocess(examples, tokenizer, max_seq_length):
    """
    Tokenize a list of (question, answer) pairs into model-ready input IDs.
    Each sequence is structured as:
      [BOS]
      <|im_start|>user<|im_end|>\n
      {question}
      <eot_id>\n
      <|im_start|>assistant<|im_end|>\n
      {answer} + EOS padding up to max_seq_length
    """
    all_input_ids = []
    prompt_lengths = []

    for question, answer in zip(examples["question"], examples["answer"]):
        prompt = [
            {"role": "user", "content": question},
        ]
        prompt_tokens = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=True
        )
        prompt_length = len(prompt_tokens)

        # 2) Tokenize the answer and prepare EOS padding
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        slots = max_seq_length - prompt_length
        if slots <= 0:
            logger.warning(
                f"max_seq_length={max_seq_length} is too small for prompt_length={prompt_length}."
            )
            continue

        #    a) Truncate answer if it’s too long
        # truncated = answer_tokens[:slots]
        #    b) Pad the rest with EOS tokens
        # eos_padding = [tokenizer.eos_token_id] * (slots - len(truncated))
        answer_section = answer_tokens  # truncated #+ eos_padding

        # 3) Combine prompt and answer into a single input_ids sequence
        input_ids = prompt_tokens + answer_section
        all_input_ids.append(input_ids)
        prompt_lengths.append(prompt_length)

    return {
        "input_ids": all_input_ids,
        "prompt_lengths": prompt_lengths,
    }
