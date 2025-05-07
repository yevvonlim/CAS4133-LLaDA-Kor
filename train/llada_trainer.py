from transformers import Trainer
import torch
import torch.nn.functional as F


class LLaDaTrainerSFT(Trainer):
    # from https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md
    def forward_process(self, input_ids, eps=1e-3):
        b, l = input_ids.shape
        t = torch.rand(b, device=input_ids.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        # 126336 is used for [MASK] token
        noisy_batch = torch.where(masked_indices, 126336, input_ids)
        return noisy_batch, masked_indices, p_mask

    # https://github.com/Orolol/gptoughts/blob/main/llada.md
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # 1) input ids
        input_ids = inputs["input_ids"]
        prompt_lengths = inputs["prompt_lengths"]  # this is the length of the prompt;
        # 2) inject noise
        noisy_batch, masked_indices, p_mask = self.forward_process(input_ids)

        # Do not add noise to the prompt part - keep it clean
        for i in range(input_ids.shape[0]):
            prompt_mask = (
                torch.arange(noisy_batch.shape[1], device=noisy_batch.device)
                < prompt_lengths[i]
            )
            noisy_batch[i, prompt_mask] = input_ids[i, prompt_mask]

        # Calculate the answer length (including padded <EOS> tokens)
        answer_lengths = input_ids.shape[1] - prompt_lengths

        # 3) forward pass
        outputs = model(
            input_ids=noisy_batch, attention_mask=inputs.get("attention_mask", None)
        )
        logits = outputs.logits  # (b, l, vocab_size)

        # Only calculate loss on masked tokens in the response
        res = ~torch.stack(
            [
                torch.arange(noisy_batch.shape[1], device=noisy_batch.device)
                < prompt_lengths[i, None]
                for i in range(input_ids.shape[0])
            ]
        )

        response_mask = masked_indices & res

        # Compute loss on masked response tokens only
        token_loss = (
            F.cross_entropy(
                logits.view(-1, logits.size(-1))[response_mask.view(-1)],
                input_ids.view(-1)[response_mask.view(-1)],
                reduction="none",
            )
            / p_mask.view(-1)[response_mask.view(-1)]
        )

        # 5) average loss
        loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

        if return_outputs:
            return loss, outputs
        return loss
