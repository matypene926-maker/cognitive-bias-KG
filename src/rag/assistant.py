"""
src/rag/assistant.py
====================
RAG (Retrieval Augmented Generation) assistant for the Cognitive Bias KG.
Answers questions grounded exclusively in the Knowledge Graph — zero hallucination.

Pipeline:
  Question → Keyword extraction → KG retrieval (1-hop) →
  Context formatting → LLM generation → Grounded answer

LLM fallback chain:
  1. Anthropic (Claude) if ANTHROPIC_API_KEY is set
  2. OpenAI (GPT) if OPENAI_API_KEY is set
  3. Heuristic mode (no API — structured answer from triples directly)
"""

import os
import re
import logging
from rdflib import Graph, Namespace, URIRef, RDFS, RDF
from pathlib import Path

logger = logging.getLogger(__name__)

CB  = Namespace("http://cogbias-kg.edu/entity/")
CBP = Namespace("http://cogbias-kg.edu/property/")
CBO = Namespace("http://cogbias-kg.edu/ontology/")

SKIP = ["22-rdf-syntax", "rdf-schema", "2002/07/owl", "skos", "isDefinedBy"]
STOPWORDS = {"what","why","how","who","where","is","are","does","do",
             "the","a","an","of","in","on","to","for","and","or"}
RICH_PREDS = {"affects","causedBy","appearsIn","reinforces","leadsTo","exploits"}


class CognitiveBiasAssistant:
    """
    RAG assistant grounded in the Cognitive Bias Knowledge Graph.

    Usage:
        assistant = CognitiveBiasAssistant()
        result = assistant.answer("Why do social networks reinforce opinions?")
        print(result["answer"])
    """

    def __init__(
        self,
        kg_path:     str = "kg_artifacts/alignment.ttl",
        schema_path: str = "kg_artifacts/ontology.ttl",
    ):
        self.g = Graph()
        self.g.bind("cb", CB); self.g.bind("cbp", CBP); self.g.bind("cbo", CBO)

        for path in [kg_path, schema_path]:
            if Path(path).exists():
                self.g.parse(path, format="turtle")
                logger.info(f"Loaded: {path} ({len(self.g)} triples)")
            else:
                logger.warning(f"Not found: {path}")

        # Build label → URI index
        self._label_index: dict[str, URIRef] = {}
        for s, _, lbl in self.g.triples((None, RDFS.label, None)):
            self._label_index[str(lbl).lower()] = s

        logger.info(f"Assistant ready — {len(self.g)} triples | {len(self._label_index)} labels")

    def _get_label(self, uri) -> str:
        if not isinstance(uri, URIRef): return str(uri)
        for _, _, lbl in self.g.triples((uri, RDFS.label, None)):
            return str(lbl)
        return str(uri).split("/")[-1].replace("_", " ")

    def retrieve(self, question: str, max_triples: int = 15) -> list[dict]:
        """Retrieve relevant triples from the KG for a given question."""
        kws = [w for w in re.findall(r"\b[a-zA-Z][a-zA-Z\-]+\b", question.lower())
               if w not in STOPWORDS and len(w) > 2]

        # Find matching entities
        entities = []
        for kw in kws:
            for label, uri in self._label_index.items():
                if kw in label and uri not in entities:
                    entities.append(uri)

        # 1-hop expansion
        triples, seen = [], set()
        for ent in entities[:5]:
            for pred, obj in self.g.predicate_objects(ent):
                if any(ns in str(pred) for ns in SKIP): continue
                key = (str(ent), str(pred), str(obj))
                if key not in seen:
                    seen.add(key)
                    triples.append({
                        "subject":   self._get_label(ent),
                        "predicate": str(pred).split("/")[-1],
                        "object":    self._get_label(obj) if isinstance(obj, URIRef) else str(obj),
                    })
            for subj, pred in self.g.subject_predicates(ent):
                if any(ns in str(pred) for ns in SKIP): continue
                key = (str(subj), str(pred), str(ent))
                if key not in seen:
                    seen.add(key)
                    triples.append({
                        "subject":   self._get_label(subj) if isinstance(subj, URIRef) else str(subj),
                        "predicate": str(pred).split("/")[-1],
                        "object":    self._get_label(ent),
                    })

        # Score: keyword match (+2) + rich predicate (+1)
        def score(t):
            txt = f"{t['subject']} {t['object']}".lower()
            s   = sum(2 for kw in kws if kw in txt)
            if t["predicate"] in RICH_PREDS: s += 1
            return s

        return sorted(triples, key=score, reverse=True)[:max_triples]

    def _format_context(self, triples: list[dict]) -> str:
        if not triples:
            return "No relevant facts found in the Knowledge Graph."
        lines = ["Facts from the Knowledge Graph:"]
        for t in triples:
            lines.append(f"  • {t['subject']} → {t['predicate']} → {t['object']}")
        return "\n".join(lines)

    def _call_llm(self, context: str, question: str) -> str:
        sys_prompt = ("You are a KG-based assistant specialized in cognitive biases on the web. "
                      "Answer ONLY from the provided Knowledge Graph facts. "
                      "Cite the triples you used. If facts are insufficient, say so explicitly.")
        user_msg = f"{context}\n\nQuestion: {question}"

        # Try Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                r = client.messages.create(
                    model="claude-sonnet-4-6", max_tokens=512,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return r.content[0].text
            except Exception as e:
                logger.warning(f"Anthropic API failed: {e}")

        # Try OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role":"system","content":sys_prompt},
                              {"role":"user","content":user_msg}],
                    max_tokens=512,
                )
                return r.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI API failed: {e}")

        # Heuristic fallback (no LLM)
        triples = self.retrieve(question)
        if not triples:
            return "No facts found in the Knowledge Graph for this question."
        by_subj: dict[str, list[str]] = {}
        for t in triples:
            s = t["subject"]
            if s not in by_subj: by_subj[s] = []
            by_subj[s].append(f"{t['predicate']} {t['object']}")
        lines = ["Based on the Knowledge Graph:"]
        for s, rels in list(by_subj.items())[:4]:
            lines.append(f"  • {s} : {', '.join(rels[:3])}")
        lines.append("\n📌 Based on:\n" + "\n".join(
            f"• {t['subject']} → {t['predicate']} → {t['object']}" for t in triples[:6]
        ))
        lines.append("\n⚠️  Heuristic mode — set ANTHROPIC_API_KEY for richer answers.")
        return "\n".join(lines)

    def answer(self, question: str) -> dict:
        """Full RAG pipeline: Retrieve → Context → Generate."""
        triples  = self.retrieve(question)
        context  = self._format_context(triples)
        response = self._call_llm(context, question)
        return {"question": question, "answer": response, "triples": triples}

    def print_answer(self, question: str) -> None:
        """Answer and print formatted output."""
        result = self.answer(question)
        print("\n" + "─" * 60)
        print(f"❓ {result['question']}")
        print("─" * 60)
        print(f"\n💡 Answer:\n{result['answer']}")
        print(f"\n📊 Triples used ({len(result['triples'])}):")
        for t in result["triples"][:6]:
            print(f"   ({t['subject']}) →[{t['predicate']}]→ ({t['object']})")
        print("─" * 60)


EXAMPLE_QUESTIONS = [
    "Why do social networks reinforce opinions?",
    "What causes echo chambers?",
    "How does the recommendation algorithm affect confirmation bias?",
    "What is the relationship between filter bubbles and fake news?",
    "What leads to algorithmic radicalization?",
    "Which platforms spread misinformation?",
    "What are the effects of political polarization?",
]


def run_demo() -> None:
    """Interactive chatbot demo."""
    print("\n" + "═" * 60)
    print("  COGNITIVE BIAS KG ASSISTANT — RAG Demo")
    print("═" * 60)
    assistant = CognitiveBiasAssistant()
    print(f"✓ KG loaded: {len(assistant.g)} triples\n")
    print("Type a question, 'examples' to see suggestions, or 'quit' to exit.\n")

    while True:
        try:
            q = input("❓ Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q: continue
        if q.lower() in ("quit","exit","q"): break
        if q.lower() == "examples":
            for i, ex in enumerate(EXAMPLE_QUESTIONS, 1):
                print(f"  {i}. {ex}")
            continue
        if q.isdigit() and 1 <= int(q) <= len(EXAMPLE_QUESTIONS):
            q = EXAMPLE_QUESTIONS[int(q)-1]
        assistant.print_answer(q)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_demo()
