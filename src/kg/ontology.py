"""
src/kg/ontology.py
==================
RDFS/OWL ontology for the Cognitive Bias Knowledge Graph.
Defines class hierarchy, property domain/range, symmetric/inverse properties.

Lab reference: TP RDFLib Exercise 4 + TP Protégé Part 2

Output: kg_artifacts/ontology.ttl
"""

import logging
from pathlib import Path
from rdflib import Graph, Namespace, Literal, RDF, RDFS, OWL

logger = logging.getLogger(__name__)

CB  = Namespace("http://cogbias-kg.edu/entity/")
CBP = Namespace("http://cogbias-kg.edu/property/")
CBO = Namespace("http://cogbias-kg.edu/ontology/")

CLASSES = [
    (CBO["PsychologicalPhenomenon"], "Psychological Phenomenon", "Any psychological phenomenon",           OWL.Thing),
    (CBO["CognitiveBias"],           "Cognitive Bias",           "Systematic deviation from rationality",  CBO["PsychologicalPhenomenon"]),
    (CBO["PerceptualBias"],          "Perceptual Bias",          "Bias in information perception",         CBO["CognitiveBias"]),
    (CBO["MemoryBias"],              "Memory Bias",              "Bias related to memory and recall",      CBO["CognitiveBias"]),
    (CBO["SocialBias"],              "Social Bias",              "Bias in social dynamics",                CBO["CognitiveBias"]),
    (CBO["Heuristic"],               "Heuristic",                "Mental shortcut for decision-making",    CBO["PsychologicalPhenomenon"]),
    (CBO["DigitalPlatform"],         "Digital Platform",         "An online platform or service",          OWL.Thing),
    (CBO["Platform"],                "Platform",                 "Alias for DigitalPlatform",              CBO["DigitalPlatform"]),
    (CBO["SocialMediaPlatform"],     "Social Media Platform",    "Platform for social content sharing",    CBO["DigitalPlatform"]),
    (CBO["SearchEngine"],            "Search Engine",            "Information retrieval system",           CBO["DigitalPlatform"]),
    (CBO["WebEffect"],               "Web Effect",               "Effect observed on the web",             OWL.Thing),
    (CBO["Effect"],                  "Effect",                   "Alias for WebEffect",                    CBO["WebEffect"]),
    (CBO["SocialEffect"],            "Social Effect",            "Effect on social dynamics",              CBO["WebEffect"]),
    (CBO["InformationEffect"],       "Information Effect",       "Effect on information quality",          CBO["WebEffect"]),
    (CBO["CognitivePhenomenon"],     "Cognitive Phenomenon",     "Large-scale cognitive effect",           CBO["WebEffect"]),
    (CBO["HumanBehavior"],           "Human Behavior",           "Observable human behavior",              OWL.Thing),
    (CBO["Behavior"],                "Behavior",                 "Alias for HumanBehavior",                CBO["HumanBehavior"]),
    (CBO["OnlineBehavior"],          "Online Behavior",          "Behavior in online contexts",            CBO["HumanBehavior"]),
    (CBO["Concept"],                 "Concept",                  "Abstract digital concept",               OWL.Thing),
    (CBO["Person"],                  "Person",                   "A person",                               OWL.Thing),
    (CBO["Organization"],            "Organization",             "An organization",                        OWL.Thing),
]

PROPS = [
    (CBP["affects"],    "affects",     CBO["CognitiveBias"],   OWL.Thing,            "A bias affects a platform, effect or behavior"),
    (CBP["causedBy"],   "caused by",   CBO["WebEffect"],       OWL.Thing,            "An effect is caused by"),
    (CBP["appearsIn"],  "appears in",  CBO["CognitiveBias"],   CBO["DigitalPlatform"],"A bias manifests on a platform"),
    (CBP["reinforces"], "reinforces",  OWL.Thing,              OWL.Thing,            "Mutual reinforcement"),
    (CBP["leadsTo"],    "leads to",    OWL.Thing,              CBO["WebEffect"],     "Leads to an effect"),
    (CBP["promotes"],   "promotes",    OWL.Thing,              OWL.Thing,            "Promotes another entity"),
    (CBP["spreads"],    "spreads",     OWL.Thing,              OWL.Thing,            "Spreads content or information"),
    (CBP["creates"],    "creates",     OWL.Thing,              OWL.Thing,            "Creates an entity or effect"),
    (CBP["influences"], "influences",  OWL.Thing,              OWL.Thing,            "Influences another entity"),
    (CBP["drives"],     "drives",      OWL.Thing,              OWL.Thing,            "Drives a behavior or outcome"),
    (CBP["exploits"],   "exploits",    CBO["DigitalPlatform"], CBO["CognitiveBias"], "Platform exploits a bias"),
    (CBP["uses"],       "uses",        CBO["DigitalPlatform"], CBO["Concept"],       "Platform uses a technology"),
    (CBP["relatedTo"],  "related to",  OWL.Thing,              OWL.Thing,            "Generic relation"),
    (CBP["causes"],     "causes",      OWL.Thing,              OWL.Thing,            "Inverse of causedBy"),
]


def build_ontology() -> Graph:
    g = Graph()
    for ns, nm in [("cbo",CBO),("cbp",CBP),("owl",OWL),("rdfs",RDFS)]:
        g.bind(ns, nm)

    for cls, label, comment, parent in CLASSES:
        g.add((cls, RDF.type,        OWL.Class))
        g.add((cls, RDFS.label,      Literal(label,   lang="en")))
        g.add((cls, RDFS.comment,    Literal(comment, lang="en")))
        g.add((cls, RDFS.subClassOf, parent))

    # Disjoint classes
    for a, b in [(CBO["PerceptualBias"],CBO["SocialBias"]),
                 (CBO["SocialMediaPlatform"],CBO["SearchEngine"]),
                 (CBO["SocialEffect"],CBO["InformationEffect"])]:
        g.add((a, OWL.disjointWith, b))

    for prop, label, domain, rng, comment in PROPS:
        g.add((prop, RDF.type,     OWL.ObjectProperty))
        g.add((prop, RDFS.label,   Literal(label,   lang="en")))
        g.add((prop, RDFS.comment, Literal(comment, lang="en")))
        g.add((prop, RDFS.domain,  domain))
        g.add((prop, RDFS.range,   rng))

    # Special property types
    g.add((CBP["reinforces"], RDF.type, OWL.SymmetricProperty))
    g.add((CBP["relatedTo"],  RDF.type, OWL.SymmetricProperty))
    g.add((CBP["relatedTo"],  RDF.type, OWL.TransitiveProperty))
    g.add((CBP["causedBy"],   OWL.inverseOf, CBP["causes"]))

    # Known instances → precise classes
    known = [
        (CB["Confirmation_Bias"],      CBO["PerceptualBias"]),
        (CB["Dunning_Kruger_Effect"],  CBO["PerceptualBias"]),
        (CB["Availability_Heuristic"], CBO["Heuristic"]),
        (CB["Echo_Chamber"],           CBO["CognitivePhenomenon"]),
        (CB["Filter_Bubble"],          CBO["CognitivePhenomenon"]),
        (CB["Bandwagon_Effect"],       CBO["SocialBias"]),
        (CB["Tribalism"],              CBO["SocialBias"]),
        (CB["Political_Polarization"], CBO["SocialEffect"]),
        (CB["Misinformation"],         CBO["InformationEffect"]),
        (CB["Fake_News"],              CBO["InformationEffect"]),
        (CB["Facebook"],               CBO["SocialMediaPlatform"]),
        (CB["Twitter"],                CBO["SocialMediaPlatform"]),
        (CB["YouTube"],                CBO["SocialMediaPlatform"]),
        (CB["Google"],                 CBO["SearchEngine"]),
    ]
    for ent, cls in known:
        g.add((ent, RDF.type, cls))

    return g


def run_ontology_builder(output_dir: str = "kg_artifacts") -> Graph:
    Path(output_dir).mkdir(exist_ok=True)
    g = build_ontology()
    g.serialize(f"{output_dir}/ontology.ttl", format="turtle")
    g.serialize(f"{output_dir}/ontology.rdf", format="xml")
    classes = sum(1 for _ in g.subjects(RDF.type, OWL.Class))
    props   = sum(1 for _ in g.subjects(RDF.type, OWL.ObjectProperty))
    logger.info(f"Ontology: {len(g)} triples | {classes} classes | {props} properties")
    return g


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    g = run_ontology_builder()
    print(f"\n✓ Ontology: {len(g)} triples → kg_artifacts/ontology.ttl")
