#!/usr/bin/env python3
"""
Download ~0.1B tokens of FineWeb raw text into fineweb_raw/.

Uses streaming to avoid downloading the full 10BT sample. Writes one .jsonl file
with one "text" field per line (and optional id for debugging). Stops when
cumulative token_count (from dataset) reaches TARGET_TOKENS.
"""

from pathlib import Path
import json

from datasets import load_dataset

# ~0.1B tokens
TARGET_TOKENS = 100_000_000
# Output dir: _dataset/data/fineweb_raw
OUT_DIR = Path(__file__).resolve().parent / "fineweb_raw"
RAW_FILE = OUT_DIR / "train.jsonl"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Streaming FineWeb (sample-10BT) until ~{TARGET_TOKENS / 1e6:.1f}M tokens...")
    print(f"Writing raw text to {OUT_DIR}")

    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    total_tokens = 0
    num_docs = 0
    with open(RAW_FILE, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text")
            if not text or not str(text).strip():
                continue
            token_count = row.get("token_count") or 0
            total_tokens += token_count
            num_docs += 1
            # One JSON object per line: at least "text"; optional "id" for traceability
            rec = {"text": text}
            if "id" in row:
                rec["id"] = row["id"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if total_tokens >= TARGET_TOKENS:
                break
            if num_docs % 50_000 == 0 and num_docs > 0:
                print(f"  docs={num_docs}, tokens≈{total_tokens / 1e6:.2f}M")

    print(f"Done: {num_docs} documents, ~{total_tokens} tokens -> {RAW_FILE}")


if __name__ == "__main__":
    main()
