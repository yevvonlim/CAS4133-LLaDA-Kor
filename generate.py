import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers import LlamaModel
from peft import PeftModel


def add_gumbel_noise(
    logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """
    Apply Gumbel noise to logits for sampling.

    Args:
        logits (torch.Tensor): Unnormalized log-probabilities.
        temperature (float): Sampling temperature. If 0, returns logits unchanged.

    Returns:
        torch.Tensor: Noisy scores ready for sampling.
    """
    if temperature <= 0:
        return logits

    # Sample standard Gumbel noise
    gumbel = -torch.log(-torch.log(torch.rand_like(logits, dtype=torch.float64)))
    noisy_logits = logits.to(torch.float64) + temperature * gumbel
    return noisy_logits.to(logits.dtype)


def get_num_transfer_tokens(
    mask_index: torch.Tensor, steps: int
) -> torch.Tensor:
    """
    Compute how many masked tokens to update per inner step.

    Args:
        mask_index (torch.Tensor): Boolean tensor indicating masked positions (batch, seq_len).
        steps (int): Total number of refinement steps per block.

    Returns:
        torch.Tensor: Tensor of shape (batch, steps) with counts per step.
    """
    batch_size = mask_index.size(0)
    total_masks = mask_index.sum(dim=1)
    base = total_masks // steps
    rem = total_masks % steps

    counts = base.unsqueeze(-1).expand(-1, steps).clone()
    for i in range(batch_size):
        counts[i, :rem[i]] += 1

    return counts


@torch.no_grad()
def generate_stream(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_token_id: int = 126336,
):
    """
    Yields intermediate token sequences while iteratively filling masked positions.

    Args:
        model (torch.nn.Module): Language model with .logits output.
        prompt_ids (torch.Tensor): Input IDs of shape (batch, prompt_len).
        steps (int): Total refinement steps per block.
        gen_length (int): Number of tokens to generate.
        block_length (int): Block size for progressive generation.
        temperature (float): Gumbel noise temperature.
        cfg_scale (float): Classifier-free guidance scale.
        remasking (str): "low_confidence" or "random".
        mask_token_id (int): Token ID for masking.

    Yields:
        torch.Tensor: Current sequence tensor of shape (batch, prompt_len + gen_length).
    """
    device = model.device
    batch_size, prompt_len = prompt_ids.shape
    total_len = prompt_len + gen_length

    # Initialize sequence with masks
    seq = torch.full(
        (batch_size, total_len), mask_token_id, dtype=torch.long, device=device
    )
    seq[:, :prompt_len] = prompt_ids
    fixed = seq != mask_token_id

    assert gen_length % block_length == 0, "gen_length must be multiple of block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    inner_steps = steps // num_blocks

    for block in range(num_blocks):
        start = prompt_len + block * block_length
        end = start + block_length
        block_mask = seq[:, start:end] == mask_token_id
        transfer_counts = get_num_transfer_tokens(block_mask, inner_steps)

        for step in range(inner_steps):
            mask_positions = seq == mask_token_id

            # Classifier-free guidance
            if cfg_scale > 0:
                uncond_seq = seq.clone()
                uncond_seq[fixed] = mask_token_id
                inputs = torch.cat([seq, uncond_seq], dim=0)
                logits = model(inputs).logits
                cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
            else:
                logits = model(seq).logits

            # Sample tokens
            noisy = add_gumbel_noise(logits, temperature)
            sampled = noisy.argmax(dim=-1)

            # Compute remasking confidence
            if remasking == "low_confidence":
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                sel_p = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                sel_p = torch.rand_like(sampled, dtype=torch.float64)
            else:
                raise ValueError(f"Unknown remasking mode: {remasking}")

            # Prevent updates outside current block
            sel_p[:, end:] = float("-inf")

            # Keep original tokens
            candidate = torch.where(mask_positions, sampled, seq)
            confidence = torch.where(mask_positions, sel_p, float("-inf"))

            # Select positions to update
            update_mask = torch.zeros_like(seq, dtype=torch.bool)
            for i in range(batch_size):
                topk_idx = confidence[i].topk(int(transfer_counts[i, step]))[1]
                update_mask[i, topk_idx] = True

            seq = torch.where(update_mask, candidate, seq)
            yield seq.clone()


if __name__ == "__main__":
    device = "cuda"
    model_path = "sionic-ai/sionic-dllm-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()

    prompt = "6나누기 0은 뭐야? let's think step by step."
    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer(chat_input, return_tensors="pt").input_ids.to(device)

    for _ in generate_stream(
        model,
        prompt_ids,
        steps=128,
        gen_length=128,
        block_length=16,
        temperature=0.0,
        cfg_scale=0.0,
    ):
        pass  # consume stream for final output

    # Decode and print final result
    final_ids = _[0, prompt_ids.shape[1]:]
    print(tokenizer.decode(final_ids, skip_special_tokens=True))
