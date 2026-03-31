"""
pipeline.py
===========
End-to-end pipeline launcher for the Cognitive Bias Knowledge Graph project.

Usage:
  python pipeline.py                  # Full pipeline
  python pipeline.py --skip-crawl     # Skip crawling (use existing data)
  python pipeline.py --rag-only       # Launch RAG assistant only
  python pipeline.py --no-rag         # Pipeline without interactive RAG
"""

import sys
import time
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  COGNITIVE BIAS KNOWLEDGE GRAPH — End-to-End Pipeline        ║
║  Web Mining & Semantics                                      ║
╠══════════════════════════════════════════════════════════════╣
║  Domain  : Cognitive Biases on the Web                       ║
║  Pipeline: Web → Text → KG → RAG Assistant                   ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_step(name: str, fn, *args, **kwargs):
    print(f"\n{'═'*60}\n▶  {name}\n{'─'*60}")
    start = time.time()
    try:
        result = fn(*args, **kwargs)
        print(f"✓  Done in {time.time()-start:.1f}s")
        return result
    except Exception as e:
        print(f"✗  Failed: {e}")
        logger.error(f"{name} failed", exc_info=True)
        return None


def run_full_pipeline(skip_crawl: bool = False, launch_rag: bool = True):
    print(BANNER)
    Path("data").mkdir(exist_ok=True)
    Path("kg_artifacts").mkdir(exist_ok=True)
    Path("data/kge").mkdir(exist_ok=True)

    #  Crawling
    if skip_crawl and Path("data/crawler_output.jsonl").exists():
        print("\n⏭  Crawling skipped (--skip-crawl, existing data found)")
    else:
        from src.crawl.crawler import run_crawler
        run_step("① WEB CRAWLING", run_crawler)

    #  Preprocessing
    if Path("data/crawler_output.jsonl").exists():
        from src.ie.preprocessing import run_preprocessing
        run_step("② PREPROCESSING", run_preprocessing)

    #  NER + Relations
    if Path("data/preprocessed.jsonl").exists():
        from src.ie.ner_extraction import run_ner_pipeline
        run_step("③ NER + RELATION EXTRACTION", run_ner_pipeline)

    #  Knowledge Graph
    from src.kg.kg_builder import run_kg_builder
    run_step("④ KNOWLEDGE GRAPH (RDF)", run_kg_builder)

    #  Ontology
    from src.kg.ontology import run_ontology_builder
    run_step("⑤ ONTOLOGY (RDFS/OWL)", run_ontology_builder)

    #  SWRL Inference
    from src.reason.swrl_rules import run_reasoning
    run_step("⑥ SWRL INFERENCE", run_reasoning)

    #  KGE
    from src.kge.embedding import run_kge_pipeline
    run_step("⑦ KNOWLEDGE GRAPH EMBEDDING", run_kge_pipeline)

    # Summary
    print("\n" + "═"*60)
    print("PIPELINE COMPLETE")
    print("═"*60)
    files = [
        ("data/crawler_output.jsonl",    "Crawled pages"),
        ("data/preprocessed.jsonl",      "Preprocessed text"),
        ("data/extracted_entities.csv",  "Extracted entities"),
        ("data/extracted_relations.csv", "Extracted relations"),
        ("kg_artifacts/biases.ttl",      "Knowledge Graph (Turtle)"),
        ("kg_artifacts/expanded.nt",     "Knowledge Graph (N-Triples)"),
        ("kg_artifacts/ontology.ttl",    "Ontology (RDFS/OWL)"),
        ("kg_artifacts/alignment.ttl",   "Inferred graph"),
        ("data/kge/train.txt",           "KGE train split"),
    ]
    for path, desc in files:
        status = "✓" if Path(path).exists() else "✗"
        size   = f"({Path(path).stat().st_size/1024:.1f} KB)" if Path(path).exists() else ""
        print(f"  {status} {desc:<30} {path} {size}")

    if launch_rag:
        print("\n\nLaunching RAG assistant...")
        time.sleep(1)
        from src.rag.assistant import run_demo
        run_demo()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognitive Bias KG — End-to-End Pipeline")
    parser.add_argument("--skip-crawl", action="store_true", help="Skip web crawling")
    parser.add_argument("--rag-only",   action="store_true", help="Launch RAG assistant only")
    parser.add_argument("--no-rag",     action="store_true", help="Pipeline without interactive RAG")
    args = parser.parse_args()

    if args.rag_only:
        from src.rag.assistant import run_demo
        run_demo()
    else:
        run_full_pipeline(
            skip_crawl=args.skip_crawl,
            launch_rag=not args.no_rag,
        )
