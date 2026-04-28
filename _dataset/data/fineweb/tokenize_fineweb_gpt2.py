#!/usr/bin/env python3
"""
Tokenize FineWeb raw text (in fineweb_raw/) with GPT-2 tokenizer and save to fineweb_tok/gpt2/.

Reads train.jsonl from fineweb_raw, produces input_ids, attention_mask, labels (same as input_ids
for LM). Uses fixed max_length and padding so outputs are ready for batching.
"""

import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

# Paths relative to this script's directory
DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "fineweb_raw"
RAW_FILE = RAW_DIR / "train.jsonl"
OUT_DIR = DATA_DIR / "fineweb_tok" / "gpt2"

MAX_LENGTH = 2048
BATCH_SIZE = 1000  # texts per tokenizer batch
CHUNK_SIZE = 10_000  # docs per parquet shard (keeps memory bounded)
TOKENIZER_NAME = "gpt2"


def load_raw_texts(path: Path):
    """Yield text from jsonl (one JSON per line, key 'text')."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text")
            if text and str(text).strip():
                yield text


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_FILE}. Run download_fineweb_raw.py first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(examples):
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors=None,
        )
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    chunk, shard = [], 0
    total_docs = 0
    print(f"Tokenizing from {RAW_FILE} with {TOKENIZER_NAME} (max_length={MAX_LENGTH}) ...")
    for text in load_raw_texts(RAW_FILE):
        chunk.append(text)
        if len(chunk) >= CHUNK_SIZE:
            ds = Dataset.from_dict({"text": chunk})
            tokenized = ds.map(
                tokenize_batch,
                batched=True,
                batch_size=BATCH_SIZE,
                remove_columns=["text"],
                desc=f"Shard {shard}",
            )
            out_file = OUT_DIR / f"train-{shard:05d}.parquet"
            tokenized.to_parquet(out_file, index=False)
            total_docs += len(chunk)
            print(f"  Wrote {out_file.name} ({total_docs} docs)")
            chunk, shard = [], shard + 1
    if chunk:
        ds = Dataset.from_dict({"text": chunk})
        tokenized = ds.map(
            tokenize_batch,
            batched=True,
            batch_size=BATCH_SIZE,
            remove_columns=["text"],
            desc=f"Shard {shard}",
        )
        out_file = OUT_DIR / f"train-{shard:05d}.parquet"
        tokenized.to_parquet(out_file, index=False)
        total_docs += len(chunk)
        print(f"  Wrote {out_file.name} ({total_docs} docs)")
    print(f"Done. Tokenized data in {OUT_DIR} ({total_docs} documents).")


if __name__ == "__main__":
    main()
