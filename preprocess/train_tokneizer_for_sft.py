"""
Re‑trains a tokenizer on a domain corpus and writes it to `--output_dir`.

Usage
-----
python tokenizer_training.py \
    --dataset_path 
    --output_dir   
"""
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from loguru import logger


def prepare_tokenizer(model_name_or_path: str):
    """Load the *base* tokenizer that will supply special tokens etc."""
    return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


def training_corpus_iter(dataset, chunk_size: int = 1000):
    """Yield slices of text so that `train_new_from_iterator` streams efficiently."""
    for i in range(0, len(dataset), chunk_size):
        texts = []
        for q, a in zip(dataset[i : i + chunk_size]["question"], dataset[i : i + chunk_size]["answer"]):
            text = q + "\n" + a
            texts.append(text)
        yield texts


def train_tokenizer(base_tokenizer, dataset_path: str, output_dir: str, vocab_size: int = 30_000):
    dataset = load_dataset("json", data_files={"train": dataset_path})["train"]
    logger.info(f"Loaded dataset at {dataset_path} → {len(dataset):,} rows")

    logger.info("Training tokenizer …")
    new_tokenizer = base_tokenizer.train_new_from_iterator(
        training_corpus_iter(dataset), vocab_size=vocab_size
    )

    # fuse tokenizer
    old_vocab      = set(base_tokenizer.get_vocab())   
    new_vocab      = set(new_tokenizer.get_vocab())
    tokens_to_add  = [t for t in new_vocab if t not in old_vocab]

    base_tokenizer.add_tokens(tokens_to_add)
    base_tokenizer.save_pretrained(output_dir)
    logger.success(f"Tokenizer written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default="korean_tokenizer")
    args = parser.parse_args()

    tokenizer = prepare_tokenizer(args.base_model)
    train_tokenizer(tokenizer, args.dataset_path, args.output_dir)