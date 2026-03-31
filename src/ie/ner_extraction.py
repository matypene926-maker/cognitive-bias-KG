"""
src/ie/ner_extraction.py
========================
Information Extraction pipeline for the Cognitive Bias KG.
Transforms cleaned text into structured facts (entities + relations).

"Text may contain knowledge, but it is not knowledge." — CM

Two-layer NER:
  1. Custom domain dictionary (COGNITIVE_BIAS, PLATFORM, EFFECT, BEHAVIOR)
  2. spaCy dependency parsing for relation extraction

Output:
  data/extracted_entities.csv
  data/extracted_relations.csv
"""

import spacy
import csv
import json
import re
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

INPUT_FILE       = "data/preprocessed.jsonl"
OUTPUT_ENTITIES  = "data/extracted_entities.csv"
OUTPUT_RELATIONS = "data/extracted_relations.csv"
MAX_CHUNK        = 8_000

# ── Domain entity dictionary ────────────────────────────────────────────────
DOMAIN_ENTITIES: dict[str, str] = {
    # Cognitive biases
    "confirmation bias":       "COGNITIVE_BIAS",
    "echo chamber":            "COGNITIVE_BIAS",
    "filter bubble":           "COGNITIVE_BIAS",
    "dunning-kruger effect":   "COGNITIVE_BIAS",
    "availability heuristic":  "COGNITIVE_BIAS",
    "anchoring bias":          "COGNITIVE_BIAS",
    "bandwagon effect":        "COGNITIVE_BIAS",
    "framing effect":          "COGNITIVE_BIAS",
    "in-group bias":           "COGNITIVE_BIAS",
    "illusory truth effect":   "COGNITIVE_BIAS",
    "motivated reasoning":     "COGNITIVE_BIAS",
    "tribalism":               "COGNITIVE_BIAS",
    "groupthink":              "COGNITIVE_BIAS",
    # Platforms
    "facebook":                "PLATFORM",
    "twitter":                 "PLATFORM",
    "youtube":                 "PLATFORM",
    "tiktok":                  "PLATFORM",
    "instagram":               "PLATFORM",
    "social media":            "PLATFORM",
    "recommendation algorithm":"PLATFORM",
    "news feed":               "PLATFORM",
    # Effects
    "polarization":            "EFFECT",
    "political polarization":  "EFFECT",
    "radicalization":          "EFFECT",
    "misinformation":          "EFFECT",
    "disinformation":          "EFFECT",
    "fake news":               "EFFECT",
    "hate speech":             "EFFECT",
    "extremism":               "EFFECT",
    "opinion formation":       "EFFECT",
    # Behaviors
    "selective exposure":      "BEHAVIOR",
    "sharing":                 "BEHAVIOR",
    "engagement":              "BEHAVIOR",
    "fact-checking":           "BEHAVIOR",
}

RELATION_VERBS = {
    "affect", "cause", "reinforce", "amplify", "create", "produce",
    "lead", "trigger", "promote", "spread", "influence", "drive",
}


def load_spacy() -> spacy.language.Language:
    for model in ["en_core_web_trf", "en_core_web_lg", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model)
            logger.info(f"spaCy model loaded: {model}")
            return nlp
        except OSError:
            continue
    raise RuntimeError("No spaCy model found. Run: python -m spacy download en_core_web_sm")


def detect_domain_entities(text: str, url: str) -> list[dict]:
    """Custom NER via dictionary matching."""
    text_lower = text.lower()
    found, seen = [], set()
    for term, label in DOMAIN_ENTITIES.items():
        if re.search(r"\b" + re.escape(term) + r"\b", text_lower) and term not in seen:
            seen.add(term)
            found.append({"entity_text": term.title(), "entity_label": label, "source_url": url})
    return found


def extract_relations(doc_spacy, domain_ents: list[dict], url: str) -> list[dict]:
    """Extract relations via dependency parsing + co-occurrence."""
    rels   = []
    active = {e["entity_text"].lower(): e["entity_label"] for e in domain_ents}

    # Dependency parsing: nsubj + VERB + dobj
    for token in doc_spacy:
        if token.lemma_.lower() not in RELATION_VERBS:
            continue
        subj = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if not subj:
            continue
        subj_match = next((t for t in active if t in subj.text.lower()), None)
        if not subj_match:
            continue
        for child in token.children:
            if child.dep_ in ("dobj", "attr"):
                obj_match = next((t for t in active if t in child.text.lower()), None)
                if obj_match:
                    rels.append({
                        "subject":       subj_match.title(),
                        "subject_label": active[subj_match],
                        "relation":      token.lemma_.lower(),
                        "object":        obj_match.title(),
                        "object_label":  active[obj_match],
                        "source_url":    url,
                    })

    # Co-occurrence fallback
    for sent in doc_spacy.sents:
        sl      = sent.text.lower()
        present = [t for t in active if t in sl]
        if len(present) == 2 and present[0] != present[1]:
            rels.append({
                "subject":       present[0].title(),
                "subject_label": active[present[0]],
                "relation":      "relatedTo",
                "object":        present[1].title(),
                "object_label":  active[present[1]],
                "source_url":    url,
            })
    return rels


def run_ner_pipeline(
    input_file:       str = INPUT_FILE,
    output_entities:  str = OUTPUT_ENTITIES,
    output_relations: str = OUTPUT_RELATIONS,
) -> tuple[list, list]:
    """Full NER + relation extraction pipeline."""
    if not Path(input_file).exists():
        raise FileNotFoundError(f"{input_file} not found. Run preprocessing first.")

    nlp  = load_spacy()
    docs = [json.loads(l) for l in open(input_file, encoding="utf-8")]

    all_entities, all_relations = [], []

    for i, doc in enumerate(docs):
        text     = doc.get("clean_text", doc["text"])
        url      = doc["url"]
        dom_ents = detect_domain_entities(text, url)
        all_entities.extend(dom_ents)

        for chunk in [text[j:j+MAX_CHUNK] for j in range(0, len(text), MAX_CHUNK)]:
            sdoc = nlp(chunk)
            all_relations.extend(extract_relations(sdoc, dom_ents, url))

        logger.info(f"[{i+1:02d}] {doc['title'][:50]} → {len(dom_ents)} entities")

    # Save entities
    with open(output_entities, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["entity_text", "entity_label", "source_url"])
        w.writeheader(); w.writerows(all_entities)

    # Save relations
    with open(output_relations, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject","subject_label","relation","object","object_label","source_url"])
        w.writeheader(); w.writerows(all_relations)

    # Stats
    lc = defaultdict(int)
    for e in all_entities: lc[e["entity_label"]] += 1
    logger.info(f"NER done: {len(all_entities)} entities | {len(all_relations)} relations")
    for label, count in sorted(lc.items(), key=lambda x: -x[1]):
        logger.info(f"  {label:<20} : {count}")

    return all_entities, all_relations


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ents, rels = run_ner_pipeline()
    print(f"\n✓ {len(ents)} entities → {OUTPUT_ENTITIES}")
    print(f"✓ {len(rels)} relations → {OUTPUT_RELATIONS}")
