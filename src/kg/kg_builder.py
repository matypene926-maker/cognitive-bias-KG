"""
src/kg/kg_builder.py
====================
RDF Knowledge Graph construction for the Cognitive Bias project.
Transforms extracted facts (CSV) into a structured RDF graph.

Namespaces:
  cb:  http://cogbias-kg.edu/entity/    → domain entities
  cbp: http://cogbias-kg.edu/property/  → semantic properties
  cbo: http://cogbias-kg.edu/ontology/  → OWL classes

Output:
  kg_artifacts/biases.ttl   (Turtle — human readable)
  kg_artifacts/expanded.nt  (N-Triples — for KGE)
  kg_artifacts/biases.rdf   (RDF/XML)
  kg_artifacts/biases.json  (JSON-LD)
"""

import csv
import re
import logging
from pathlib import Path
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL

logger = logging.getLogger(__name__)

CB   = Namespace("http://cogbias-kg.edu/entity/")
CBP  = Namespace("http://cogbias-kg.edu/property/")
CBO  = Namespace("http://cogbias-kg.edu/ontology/")

LABEL_TO_CLASS = {
    "COGNITIVE_BIAS": CBO["CognitiveBias"],
    "PLATFORM":       CBO["Platform"],
    "EFFECT":         CBO["Effect"],
    "BEHAVIOR":       CBO["Behavior"],
    "CONCEPT":        CBO["Concept"],
    "PERSON":         CBO["Person"],
    "ORG":            CBO["Organization"],
}

RELATION_MAP = {
    "affect":    CBP["affects"],
    "cause":     CBP["causedBy"],
    "reinforce": CBP["reinforces"],
    "amplify":   CBP["reinforces"],
    "lead":      CBP["leadsTo"],
    "promote":   CBP["promotes"],
    "spread":    CBP["spreads"],
    "create":    CBP["creates"],
    "influence": CBP["influences"],
    "drive":     CBP["drives"],
    "trigger":   CBP["triggers"],
    "relatedTo": CBP["relatedTo"],
}

# 45 high-quality manually verified seed triples
SEED_TRIPLES = [
    ("Confirmation_Bias",    CBO["CognitiveBias"], CBP["affects"],    "Opinion_Formation",        CBO["Effect"]),
    ("Confirmation_Bias",    CBO["CognitiveBias"], CBP["appearsIn"],  "Social_Media",             CBO["Platform"]),
    ("Confirmation_Bias",    CBO["CognitiveBias"], CBP["reinforces"], "Echo_Chamber",             CBO["CognitiveBias"]),
    ("Confirmation_Bias",    CBO["CognitiveBias"], CBP["leadsTo"],    "Political_Polarization",   CBO["Effect"]),
    ("Echo_Chamber",         CBO["CognitiveBias"], CBP["causedBy"],   "Recommendation_Algorithm", CBO["Concept"]),
    ("Echo_Chamber",         CBO["CognitiveBias"], CBP["appearsIn"],  "Facebook",                 CBO["Platform"]),
    ("Echo_Chamber",         CBO["CognitiveBias"], CBP["appearsIn"],  "Twitter",                  CBO["Platform"]),
    ("Echo_Chamber",         CBO["CognitiveBias"], CBP["leadsTo"],    "Political_Polarization",   CBO["Effect"]),
    ("Echo_Chamber",         CBO["CognitiveBias"], CBP["reinforces"], "Confirmation_Bias",        CBO["CognitiveBias"]),
    ("Filter_Bubble",        CBO["CognitiveBias"], CBP["causedBy"],   "Recommendation_Algorithm", CBO["Concept"]),
    ("Filter_Bubble",        CBO["CognitiveBias"], CBP["appearsIn"],  "Facebook",                 CBO["Platform"]),
    ("Filter_Bubble",        CBO["CognitiveBias"], CBP["leadsTo"],    "Misinformation",           CBO["Effect"]),
    ("Filter_Bubble",        CBO["CognitiveBias"], CBP["reinforces"], "Confirmation_Bias",        CBO["CognitiveBias"]),
    ("Algorithmic_Radicalization", CBO["Effect"],  CBP["causedBy"],   "Recommendation_Algorithm", CBO["Concept"]),
    ("Algorithmic_Radicalization", CBO["Effect"],  CBP["appearsIn"],  "YouTube",                  CBO["Platform"]),
    ("Algorithmic_Radicalization", CBO["Effect"],  CBP["leadsTo"],    "Extremism",                CBO["Effect"]),
    ("Recommendation_Algorithm",   CBO["Concept"], CBP["reinforces"], "Confirmation_Bias",        CBO["CognitiveBias"]),
    ("Recommendation_Algorithm",   CBO["Concept"], CBP["creates"],    "Filter_Bubble",            CBO["CognitiveBias"]),
    ("Recommendation_Algorithm",   CBO["Concept"], CBP["drives"],     "Engagement",               CBO["Behavior"]),
    ("Recommendation_Algorithm",   CBO["Concept"], CBP["affects"],    "Political_Polarization",   CBO["Effect"]),
    ("Fake_News",            CBO["Effect"],         CBP["spreads"],   "Social_Media",             CBO["Platform"]),
    ("Fake_News",            CBO["Effect"],         CBP["causedBy"],  "Confirmation_Bias",        CBO["CognitiveBias"]),
    ("Fake_News",            CBO["Effect"],         CBP["leadsTo"],   "Misinformation",           CBO["Effect"]),
    ("Misinformation",       CBO["Effect"],         CBP["appearsIn"], "Online_Media",             CBO["Platform"]),
    ("Misinformation",       CBO["Effect"],         CBP["causedBy"],  "Filter_Bubble",            CBO["CognitiveBias"]),
    ("Misinformation",       CBO["Effect"],         CBP["leadsTo"],   "Political_Division",       CBO["Effect"]),
    ("Dunning_Kruger_Effect",CBO["CognitiveBias"],  CBP["affects"],   "Online_Discourse",         CBO["Behavior"]),
    ("Dunning_Kruger_Effect",CBO["CognitiveBias"],  CBP["reinforces"],"Misinformation",           CBO["Effect"]),
    ("Availability_Heuristic",CBO["CognitiveBias"], CBP["affects"],   "News_Consumption",         CBO["Behavior"]),
    ("Availability_Heuristic",CBO["CognitiveBias"], CBP["leadsTo"],   "Overestimation",           CBO["Effect"]),
    ("Bandwagon_Effect",     CBO["CognitiveBias"],  CBP["affects"],   "Social_Media",             CBO["Platform"]),
    ("Bandwagon_Effect",     CBO["CognitiveBias"],  CBP["reinforces"],"Tribalism",                CBO["CognitiveBias"]),
    ("Illusory_Truth_Effect",CBO["CognitiveBias"],  CBP["reinforces"],"Misinformation",           CBO["Effect"]),
    ("Illusory_Truth_Effect",CBO["CognitiveBias"],  CBP["affects"],   "Opinion_Formation",        CBO["Effect"]),
    ("Motivated_Reasoning",  CBO["CognitiveBias"],  CBP["reinforces"],"Confirmation_Bias",        CBO["CognitiveBias"]),
    ("Motivated_Reasoning",  CBO["CognitiveBias"],  CBP["affects"],   "Selective_Exposure",       CBO["Behavior"]),
    ("Facebook",             CBO["Platform"],       CBP["uses"],      "Recommendation_Algorithm", CBO["Concept"]),
    ("YouTube",              CBO["Platform"],       CBP["uses"],      "Recommendation_Algorithm", CBO["Concept"]),
    ("Twitter",              CBO["Platform"],       CBP["spreads"],   "Misinformation",           CBO["Effect"]),
    ("TikTok",               CBO["Platform"],       CBP["drives"],    "Engagement",               CBO["Behavior"]),
    ("Political_Polarization",CBO["Effect"],        CBP["causedBy"],  "Echo_Chamber",             CBO["CognitiveBias"]),
    ("Political_Polarization",CBO["Effect"],        CBP["causedBy"],  "Filter_Bubble",            CBO["CognitiveBias"]),
    ("Political_Polarization",CBO["Effect"],        CBP["causedBy"],  "Confirmation_Bias",        CBO["CognitiveBias"]),
    ("Selective_Exposure",   CBO["Behavior"],       CBP["reinforces"],"Confirmation_Bias",        CBO["CognitiveBias"]),
    ("Tribalism",            CBO["CognitiveBias"],  CBP["leadsTo"],   "Political_Polarization",   CBO["Effect"]),
]


def uri(text: str) -> URIRef:
    return CB[re.sub(r"[^\w\-.]", "", text.strip().replace(" ", "_"))]

def rel_uri(rel: str) -> URIRef:
    return RELATION_MAP.get(rel.lower(), CBP[rel.replace(" ", "_")])


def build_graph(
    entities_csv:  str = "data/extracted_entities.csv",
    relations_csv: str = "data/extracted_relations.csv",
) -> Graph:
    """Build the RDF graph from seed triples + extracted CSV."""
    g = Graph()
    for ns, nm in [("cb",CB),("cbp",CBP),("cbo",CBO),("rdf",RDF),("rdfs",RDFS),("owl",OWL)]:
        g.bind(ns, nm)

    # Seed triples
    for sf, sc, prop, of, oc in SEED_TRIPLES:
        su, ou = CB[sf], CB[of]
        g.add((su, RDF.type, sc));   g.add((ou, RDF.type, oc))
        g.add((su, RDFS.label, Literal(sf.replace("_", " "), lang="en")))
        g.add((ou, RDFS.label, Literal(of.replace("_", " "), lang="en")))
        g.add((su, prop, ou))

    # Entities CSV
    if Path(entities_csv).exists():
        with open(entities_csv, encoding="utf-8") as f:
            seen = set()
            for row in csv.DictReader(f):
                text, label = row["entity_text"].strip(), row["entity_label"].strip()
                if text.lower() not in seen:
                    seen.add(text.lower())
                    eu = uri(text)
                    g.add((eu, RDF.type,   LABEL_TO_CLASS.get(label, CBO["Entity"])))
                    g.add((eu, RDFS.label, Literal(text, lang="en")))

    # Relations CSV
    if Path(relations_csv).exists():
        with open(relations_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                s, p, o = row["subject"], row["relation"], row["object"]
                sl, ol  = row.get("subject_label",""), row.get("object_label","")
                if s and p and o:
                    su, ou, pu = uri(s), uri(o), rel_uri(p)
                    if sl: g.add((su, RDF.type, LABEL_TO_CLASS.get(sl, CBO["Entity"])))
                    if ol: g.add((ou, RDF.type, LABEL_TO_CLASS.get(ol, CBO["Entity"])))
                    g.add((su, pu, ou))

    logger.info(f"Graph built: {len(g)} triples")
    return g


def run_kg_builder(output_dir: str = "kg_artifacts") -> Graph:
    """Build and serialize the KG in all formats."""
    Path(output_dir).mkdir(exist_ok=True)
    g = build_graph()

    g.serialize(f"{output_dir}/biases.ttl",  format="turtle")
    g.serialize(f"{output_dir}/expanded.nt", format="nt")
    g.serialize(f"{output_dir}/biases.rdf",  format="xml")
    g.serialize(f"{output_dir}/biases.json", format="json-ld", indent=2)

    entities = len(set(g.subjects(RDF.type, None)))
    preds    = len(set(g.predicates()))
    logger.info(f"KG serialized: {len(g)} triples | {entities} entities | {preds} predicates")
    return g


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    g = run_kg_builder()
    print(f"\n✓ {len(g)} triples")
    print("  kg_artifacts/biases.ttl")
    print("  kg_artifacts/expanded.nt")
    print("  kg_artifacts/biases.rdf")
    print("  kg_artifacts/biases.json")
