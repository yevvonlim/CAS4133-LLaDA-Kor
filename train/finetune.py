# from https://github.com/QwenLM/Qwen/blob/main/finetune.py


from dataclasses import dataclass, field
import json
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from train.llada_trainer import LLaDaTrainerSFT
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from preprocess.llada_preprocess import preprocess
from datasets import load_dataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def get_llada_collator(eos_token_id):
    """
    returns: collator_fn, Trainer <- data_collator=get_llada_collator()
    """

    def collator_fn(features):
        # 1) list of list -> tensor
        input_ids = [f["input_ids"].to(torch.long) for f in features]

        prompt_lengths = torch.tensor(
            [f["prompt_lengths"] for f in features], dtype=torch.long
        )
        # 2) pad to the max length in the batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=eos_token_id
        )
        return {
            "input_ids": input_ids,  # (batch, seq_len)
            "prompt_lengths": prompt_lengths,  # (batch,)
        }

    return collator_fn


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained tokenizer."}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    evaluation_strategy="steps",      # "no" | "steps" | "epoch"
    eval_steps=500,                   


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "attn_out",
            "ff_out",
            "ff_proj",
            "up_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


class LLaDaSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, data_path, tokenizer: transformers.PreTrainedTokenizer, max_len: int, phase="train"
    ):
        super(LLaDaSupervisedDataset, self).__init__()
        # self.ds = load_dataset("json", data_files={"train": data_path})["train"]
        self.ds = load_dataset(data_path)[phase]
        # rank0_print("Formatting inputs...")
        self.ds = self.ds.map(
            lambda batch: preprocess(batch, tokenizer, max_len),
            batched=True,
            batch_size=1000,
            remove_columns=self.ds.column_names,
            desc="Preprocessing",
            num_proc=128,
        )
        self.ds.set_format(
            type="torch",
            columns=["input_ids", "prompt_lengths"],
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.ds[i]


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LLaDaSupervisedDataset
    rank0_print("Loading data...")

    train_dataset = dataset_cls(
        data_args.data_path, tokenizer=tokenizer, max_len=max_len
    )

    if data_args.eval_data_path:
        eval_dataset = dataset_cls(data_args.eval_data_path, tokenizer=tokenizer, max_len=max_len, phase="validation")
    else:
        eval_dataset = None

    data_collator = get_llada_collator(tokenizer.pad_token_id)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if (
        getattr(training_args, "deepspeed", None)
        and int(os.environ.get("WORLD_SIZE", 1)) == 1
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 are incompatible with QLoRA.")

    is_chat_model = "chat" in model_args.model_name_or_path.lower()
    if (
        training_args.use_lora
        and not lora_args.q_lora
        and deepspeed.is_deepspeed_zero3_enabled()
        and not is_chat_model
    ):
        raise RuntimeError(
            "ZeRO3 is incompatible with LoRA when finetuning on base model."
        )

    model_load_kwargs = {
        "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        **model_load_kwargs,
    )
    logging.info(f"Model loaded from {model_args.model_name_or_path}")
    tokenizer_path = model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path is not None else model_args.model_name_or_path
    logging.info(f"Tokenizer loaded from {tokenizer_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        # use_fast=False,
        trust_remote_code=True,
    )
    # special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
    # tokenizer.add_special_tokens(special_tokens)
    # model.resize_token_embeddings(len(tokenizer))

    if training_args.use_lora:
        if lora_args.q_lora or is_chat_model:
            modules_to_save = None
        else:
            modules_to_save = ["wte", "ff_out"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save,  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()


    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = LLaDaTrainerSFT(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()

    trainer.model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    # safe_save_model_for_hf_trainer(
    #     trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
    # )


if __name__ == "__main__":
    train()
