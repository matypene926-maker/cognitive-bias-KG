"""
src/reason/swrl_rules.py
========================
SWRL-style inference rules for the Cognitive Bias KG.
Derives new facts not explicitly encoded in the graph.

Lab reference: TP RDFLib Exercise 6 + KB Lab Part 1

Rules:
  R1 — reinforces is SYMMETRIC
  R2 — relatedTo is TRANSITIVE  (1-hop only)
  R3 — causedBy / causes are INVERSE
  R4 — causedBy + appearsIn → appearsIn  (chain rule)
  R5 — Type inference from RDFS domain/range
  R6 — SWRL: CognitiveBias ∧ affects(Opinion_Formation) → PerceptualBias
       (direct analogy of KB Lab: "older than 60 → OldPerson")

Output:
  kg_artifacts/alignment.ttl   (inferred graph)
  data/inferred_triples.csv
"""

import csv
import logging
from pathlib import Path
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL

logger = logging.getLogger(__name__)

CB  = Namespace("http://cogbias-kg.edu/entity/")
CBP = Namespace("http://cogbias-kg.edu/property/")
CBO = Namespace("http://cogbias-kg.edu/ontology/")


def _label(uri) -> str:
    return str(uri).split("/")[-1].replace("_", " ")


def apply_rules(g: Graph) -> tuple[Graph, list[dict]]:
    """Apply all inference rules and return enriched graph + inferred list."""
    inferred = []

    def add(s, p, o, rule):
        if (s, p, o) not in g:
            g.add((s, p, o))
            inferred.append({
                "rule":      rule,
                "subject":   _label(s),
                "predicate": _label(p),
                "object":    _label(o),
            })

    # R1 — reinforces is SYMMETRIC
    # Analogy: has-friend is symmetric (TP RDFLib Ex6, Rule 1)
    for s, _, o in list(g.triples((None, CBP["reinforces"], None))):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            add(o, CBP["reinforces"], s, "R1 — Symmetry: reinforces")

    # R2 — relatedTo is TRANSITIVE (1 level only)
    # Analogy: has-friend is transitive (TP RDFLib Ex6, Rule 2)
    pairs = list(g.triples((None, CBP["relatedTo"], None)))
    for s1, _, o1 in pairs:
        for s2, _, o2 in pairs:
            if o1 == s2 and s1 != o2 and isinstance(s1, URIRef) and isinstance(o2, URIRef):
                add(s1, CBP["relatedTo"], o2, "R2 — Transitivity: relatedTo")

    # R3 — causedBy and causes are INVERSE
    # Analogy: has-sister / has-brother are inverse (TP RDFLib Ex6, Rule 3)
    for s, _, o in list(g.triples((None, CBP["causedBy"], None))):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            add(o, CBP["causes"], s, "R3 — Inverse: causedBy/causes")

    # R4 — Chain: causedBy + appearsIn → appearsIn
    # Analogy: "An enemy of a friend is an enemy" (TP RDFLib Ex6, Rule 4)
    for effect, _, bias in list(g.triples((None, CBP["causedBy"], None))):
        for _, _, platform in list(g.triples((bias, CBP["appearsIn"], None))):
            if isinstance(effect, URIRef) and isinstance(platform, URIRef):
                add(effect, CBP["appearsIn"], platform, "R4 — Chain: causedBy+appearsIn")

    # R5 — RDFS-style type inference from domain/range
    for s, _, o in list(g.triples((None, CBP["leadsTo"], None))):
        if isinstance(o, URIRef): add(o, RDF.type, CBO["Effect"], "R5 — Type: leadsTo→Effect")
    for s, _, o in list(g.triples((None, CBP["appearsIn"], None))):
        if isinstance(o, URIRef): add(o, RDF.type, CBO["Platform"], "R5 — Type: appearsIn→Platform")
    for s, _, o in list(g.triples((None, CBP["affects"], None))):
        if isinstance(s, URIRef): add(s, RDF.type, CBO["CognitiveBias"], "R5 — Type: affects→CognitiveBias")

    # R6 — SWRL rule (main rule — KB Lab Part 1 pattern)
    # CognitiveBias(?x) ∧ affects(?x, Opinion_Formation) → PerceptualBias(?x)
    # Direct analog of: Person(?p) ∧ age(?p,?a) ∧ greaterThan(?a,60) → OldPerson(?p)
    for bias, _, _ in list(g.triples((None, CBP["affects"], CB["Opinion_Formation"]))):
        if (bias, RDF.type, CBO["CognitiveBias"]) in g:
            add(bias, RDF.type, CBO["PerceptualBias"], "R6 — SWRL: affects(Opinion_Formation)→PerceptualBias")

    return g, inferred


def run_reasoning(
    kg_path:     str = "kg_artifacts/biases.ttl",
    schema_path: str = "kg_artifacts/ontology.ttl",
    output_dir:  str = "kg_artifacts",
) -> tuple[Graph, list[dict]]:
    """Full reasoning pipeline."""
    g = Graph()
    g.bind("cb", CB); g.bind("cbp", CBP); g.bind("cbo", CBO)

    for path in [kg_path, schema_path]:
        if Path(path).exists():
            g.parse(path, format="turtle")
            logger.info(f"Loaded: {path}")

    original = len(g)
    g, inferred = apply_rules(g)

    Path(output_dir).mkdir(exist_ok=True)
    g.serialize(f"{output_dir}/alignment.ttl", format="turtle")

    # Save inferred triples CSV
    Path("data").mkdir(exist_ok=True)
    with open("data/inferred_triples.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rule","subject","predicate","object"])
        w.writeheader(); w.writerows(inferred)

    logger.info(f"Inference: {len(inferred)} new triples | graph: {original} → {len(g)}")

    # Print examples grouped by rule
    shown = set()
    for t in inferred:
        if t["rule"] not in shown:
            shown.add(t["rule"])
            logger.info(f"  [{t['rule']}]")
            logger.info(f"    ({t['subject']}) --[{t['predicate']}]--> ({t['object']})")

    return g, inferred


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    g, inferred = run_reasoning()
    print(f"\n✓ {len(inferred)} inferred triples → kg_artifacts/alignment.ttl")
