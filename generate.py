import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def generate_stream(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Generator version of `generate`, yields intermediate token sequences (`x`) after each sampling step.
    """
    device = model.device
    L = prompt.shape[1]
    x = torch.full((1, L + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :L] = prompt
    prompt_index = x != mask_id

    assert gen_length % block_length == 0, "gen_length must be multiple of block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    inner_steps = steps // num_blocks

    for b in range(num_blocks):
        start = L + b * block_length
        end = start + block_length
        block_mask = x[:, start:end] == mask_id
        transfer_counts = get_num_transfer_tokens(block_mask, inner_steps)
        for i in range(inner_steps):
            mask_index = x == mask_id

            # classifier-free guidance
            if cfg_scale > 0:
                uncond = x.clone()
                uncond[prompt_index] = mask_id
                x_in = torch.cat([x, uncond], dim=0)
                logits = model(x_in).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # sampling
            noisy = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(noisy, dim=-1)

            # compute confidence for remasking
            if remasking == "low_confidence":
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                sel_p = torch.squeeze(torch.gather(probs, -1, x0.unsqueeze(-1)), -1)
            elif remasking == "random":
                sel_p = torch.rand_like(x0, dtype=torch.float64)
            else:
                raise NotImplementedError(remasking)
            # disable remasking outside current block
            sel_p[:, end:] = -np.inf

            # keep previous tokens
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, sel_p, -np.inf)

            # select positions to update
            transfer_idx = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(x0.size(0)):
                topk = torch.topk(confidence[j], k=int(transfer_counts[j, i]))[1]
                transfer_idx[j, topk] = True

            # apply update
            x = torch.where(transfer_idx, x0, x)

            # yield a clone to avoid in-place issues
            yield x.clone()

    # final output
    return

# Example usage:
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True).to('cuda').eval()
prompt_text = "Your input prompt"
prompt_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to('cuda')
for step_x in generate_stream(model, prompt_ids, steps=128, gen_length=128, block_length=32):
    decoded = tokenizer.decode(step_x[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    print(decoded, end='\r')
