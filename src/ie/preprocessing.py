"""
src/ie/preprocessing.py
=======================
Text preprocessing pipeline for the Cognitive Bias KG project.
Cleans raw crawled text and segments into sentences.

Input:  data/crawler_output.jsonl
Output: data/preprocessed.jsonl
"""

import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

INPUT_FILE  = "data/crawler_output.jsonl"
OUTPUT_FILE = "data/preprocessed.jsonl"

BOILERPLATE = [
    r"^\s*\[\s*edit\s*\]",
    r"^\s*Retrieved from",
    r"^\s*Jump to\s*(navigation|search)",
    r"^\s*(See also|References|External links|Further reading|Notes)\s*$",
    r"^\^\s*\d+",
]


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if any(re.search(p, line, re.I) for p in BOILERPLATE):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def segment_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sents if len(s.split()) >= 5]


def run_preprocessing(
    input_file:  str = INPUT_FILE,
    output_file: str = OUTPUT_FILE,
) -> list[dict]:
    if not Path(input_file).exists():
        raise FileNotFoundError(f"{input_file} not found. Run crawler first.")

    docs_in = [json.loads(l) for l in open(input_file, encoding="utf-8")]
    Path(output_file).unlink(missing_ok=True)
    docs_out = []

    for doc in docs_in:
        clean = clean_text(doc["text"])
        sents = segment_sentences(clean)
        if len(sents) < 3:
            continue
        result = {**doc, "clean_text": clean, "sentences": sents, "sentence_count": len(sents)}
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        docs_out.append(result)
        logger.info(f"  {doc['title'][:50]} — {len(sents)} sentences")

    logger.info(f"Preprocessing done: {len(docs_out)} documents → {output_file}")
    return docs_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    docs = run_preprocessing()
    print(f"\n✓ {len(docs)} documents → {OUTPUT_FILE}")
