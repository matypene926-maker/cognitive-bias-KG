#  Cognitive Bias Knowledge Graph
**Web Mining & Semantics — Academic Project**

> End-to-end pipeline from raw web pages to a knowledge-supported AI assistant  
> on the domain of **cognitive biases applied to the web**.

---

## Overview

This project implements a complete pipeline transforming Wikipedia pages about
cognitive biases (confirmation bias, echo chambers, filter bubbles, fake news,
political polarization) into a structured **RDF Knowledge Graph**, enriched with
SWRL-style inference rules and connected to a **RAG assistant** that answers
questions grounded exclusively in the graph — zero hallucination.

---

## Project Structure

```
project-root/
├── src/
│   ├── crawl/          # Web crawling (httpx + trafilatura)
│   ├── ie/             # Information extraction (spaCy NER + preprocessing)
│   ├── kg/             # RDF Knowledge Graph + RDFS/OWL ontology
│   ├── reason/         # SWRL-style inference rules
│   ├── kge/            # Knowledge Graph Embedding (TransE, DistMult)
│   └── rag/            # RAG assistant (KG → LLM → grounded answers)
├── data/               # Generated data files (JSONL, CSV, KGE splits)
│   └── samples/        # Pre-generated sample files for demo
├── kg_artifacts/       # RDF artifacts (TTL, NT, OWL)
│   ├── ontology.ttl    # RDFS/OWL ontology
│   ├── expanded.nt     # Full KG in N-Triples (for KGE)
│   └── alignment.ttl   # Inferred graph (after SWRL rules)
├── reports/            # Final report
├── notebooks/          # Jupyter notebook (cognitive_bias_KG.ipynb)
├── pipeline.py         # End-to-end launcher
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Pipeline

```
① Web Crawling     → data/crawler_output.jsonl
② Preprocessing    → data/preprocessed.jsonl
③ NER + Relations  → data/extracted_*.csv
④ Knowledge Graph  → kg_artifacts/biases.ttl (+ .nt, .rdf, .json)
⑤ Ontology         → kg_artifacts/ontology.ttl
⑥ SWRL Inference   → kg_artifacts/alignment.ttl
⑦ KGE              → data/kge/train.txt (+ valid, test)
⑧ RAG Assistant    → Interactive chatbot
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Run full pipeline
python pipeline.py

# 3. Or step by step
python -m src.crawl.crawler
python -m src.ie.preprocessing
python -m src.ie.ner_extraction
python -m src.kg.kg_builder
python -m src.kg.ontology
python -m src.reason.swrl_rules
python -m src.kge.embedding
python -m src.rag.assistant

# 4. Skip crawling (use sample data)
python pipeline.py --skip-crawl

# 5. RAG only
python pipeline.py --rag-only
```

---

## Knowledge Graph Statistics

| Metric | Value |
|--------|-------|
| Total triples | 354 |
| Seed triples | 45 |
| Inferred triples | 40 |
| Unique entities | 35 |
| Distinct predicates | 12 |
| OWL classes | 21 |
| Object properties | 14 |

---

## Inference Rules (SWRL)

| Rule | Description | Lab Analogy |
|------|-------------|-------------|
| R1 | `reinforces` is symmetric | `has-friend` symmetric |
| R2 | `relatedTo` is transitive | `has-friend` transitive |
| R3 | `causedBy` / `causes` inverse | `has-sister` / `has-brother` |
| R4 | Chain: causedBy + appearsIn → appearsIn | "enemy of a friend is an enemy" |
| R5 | RDFS type inference from domain/range | domain/range inference |
| **R6** | **CognitiveBias ∧ affects(Opinion_Formation) → PerceptualBias** | **"older than 60 → OldPerson"** |

---

## RAG Assistant Demo

```
Why do social networks reinforce opinions?

 Based on the Knowledge Graph:
  • Confirmation Bias: appearsIn Social Media, reinforces Echo Chamber
  • Recommendation Algorithm: reinforces Confirmation Bias, creates Filter Bubble
  • Echo Chamber: causedBy Recommendation Algorithm, leadsTo Political Polarization

Triples used:
   (Confirmation Bias) →[appearsIn]→ (Social Media)
   (Echo Chamber) →[causedBy]→ (Recommendation Algorithm)
   (Recommendation Algorithm) →[reinforces]→ (Confirmation Bias)
```

---

## KGE Results

| Model | MRR | Hits@1 | Hits@3 | Hits@10 |
|-------|-----|--------|--------|---------|
| TransE | 0.312 | 0.187 | 0.354 | 0.521 |
| DistMult | 0.289 | 0.163 | 0.318 | 0.487 |

TransE outperforms DistMult on this domain due to the asymmetric nature
of relations (causedBy, leadsTo, affects).

---

## Optional: LLM API for RAG

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # Claude (recommended)
export OPENAI_API_KEY="sk-..."          # GPT fallback
```

Without an API key, the assistant runs in heuristic mode —
answers are structured directly from retrieved triples, still zero hallucination.

---

## Lab References

| Step | Lab |
|------|-----|
| Web Crawling + NER | Lab Session 1 |
| RDF + formats | TP RDFLib Ex 1–3 |
| Ontology RDFS | TP RDFLib Ex 4 + TP Protégé Part 2 |
| Inference + SWRL | TP RDFLib Ex 6 + KB Lab Part 1 |
| SPARQL | TP Protégé Parts 3 & 4 |
| KGE | KB Lab Part 2 |
| RAG | Final Project |
