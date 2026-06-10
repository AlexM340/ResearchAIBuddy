"""Graph ingestion pipeline: chunk -> concepts/relations -> Neo4j upsert."""

from __future__ import annotations

import logging
import math
import re
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Tuple

from .neo4j_client import Neo4jClient


def _cosine(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

logger = logging.getLogger(__name__)

STOPWORDS = {
    "care",
    "este",
    "sunt",
    "pentru",
    "despre",
    "acest",
    "aceasta",
    "the",
    "si",
    "sau",
    "din",
    "prin",
    "fara",
    "with",
    "that",
    "this",
    "from",
    "unde",
    "cand",
    "cum",
    "iar",
    "asa",
    "deci",
    "fiind",
    "also",
}

RELATION_MARKERS = {
    "DEPENDS_ON": {"depinde de", "depends on", "bazat pe", "based on"},
    "CONTRADICTS": {"contrazice", "incompatibil", "conflict", "opune"},
    "DERIVED_FROM": {"deriva din", "provenit din", "derived from", "rezulta din"},
    "USES": {"foloseste", "utilizeaza", " uses ", " using ", " via "},
    "PART_OF": {"parte din", "face parte", "part of", "componenta a"},
    "IMPROVES": {"imbunatateste", "improves", "optimizeaza", "creste performanta"},
}

# ---------------------------------------------------------------------------
# Ontologie: tipuri de noduri si relatii permise (extractie cu schema inchisa).
# "Concept" e tipul generic / escape-hatch pentru coada lunga.
# ---------------------------------------------------------------------------
DEFAULT_NODE_TYPE = "Concept"
NODE_TYPES = {
    "Concept",
    "Method",
    "Technology",
    "Person",
    "Organization",
    "Dataset",
    "Metric",
    "Task",
    "Event",
}

# relation_type -> (tipuri sursa permise, tipuri tinta permise); None = orice tip.
# Folosit pentru validarea de schema a muchiilor (Phase 3).
RELATION_SCHEMA: Dict[str, Tuple[Optional[set], Optional[set]]] = {
    "RELATED_TO": (None, None),
    "DEPENDS_ON": (None, None),
    "CONTRADICTS": (None, None),
    "DERIVED_FROM": (None, None),
    "USES": (None, None),
    "PART_OF": (None, None),
    "IMPROVES": (None, {"Metric", "Method", "Technology", "Concept"}),
    "PROPOSED_BY": (None, {"Person", "Organization"}),
}
ALLOWED_RELATIONS = set(RELATION_SCHEMA.keys())

# Sufixe pe care NU le scurtam la singularizare (evita "analysis"->"analysi" etc.).
_PLURAL_SAFE_SUFFIXES = ("ss", "us", "is", "os", "as")


class ConceptResolver:
    """Entity resolution: mapeaza forme de suprafata la un concept canonic.

    Strategie (in ordine):
      1. normalizare lexicala (lowercase, trim, singularizare conservatoare);
      2. potrivire exacta pe nume canonic sau alias cunoscut;
      3. potrivire semantica prin embeddings (cosine >= threshold) — prinde
         variante non-lexicale ("ML" vs "machine learning").
    Degradeaza grata: fara embedder ramane doar la pasii 1-2; fara Neo4j (sau
    DummyClient) pleaca de la un index gol si trateaza totul ca nou.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        embedder: Optional[Callable[[str], Optional[List[float]]]] = None,
        threshold: float = 0.86,
        max_existing: int = 1500,
    ):
        self.client = neo4j_client
        self.embedder = embedder
        self.threshold = float(threshold)
        self.max_existing = int(max_existing)
        self._primed = False
        self._index: List[Dict[str, Any]] = []  # {canonical, type, aliases:set, vec}
        self._alias_to_canonical: Dict[str, str] = {}

    def reset(self) -> None:
        """Goleste indexul in-memory; urmatorul resolve va re-prima din Neo4j.

        Necesar dupa un wipe al grafului (rebuild) ca sa nu deduplicam fata de
        concepte deja sterse.
        """
        self._primed = False
        self._index = []
        self._alias_to_canonical = {}

    @staticmethod
    def normalize(name: str) -> str:
        s = re.sub(r"\s+", " ", (name or "").strip().lower())
        s = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)
        if not s:
            return ""
        # Singularizare conservatoare doar pe ultimul token al frazei.
        parts = s.split(" ")
        last = parts[-1]
        if (
            len(last) > 4
            and last.endswith("s")
            and not last.endswith(_PLURAL_SAFE_SUFFIXES)
        ):
            parts[-1] = last[:-1]
        return " ".join(parts)

    def _embed(self, text: str) -> Optional[List[float]]:
        if self.embedder is None or not text:
            return None
        try:
            vec = self.embedder(text)
            return list(vec) if vec is not None else None
        except Exception:
            return None

    def _register(
        self,
        canonical: str,
        node_type: str,
        aliases: Optional[set] = None,
        vec: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        entry = {
            "canonical": canonical,
            "type": node_type or DEFAULT_NODE_TYPE,
            "aliases": set(aliases or set()),
            "vec": vec,
        }
        self._index.append(entry)
        self._alias_to_canonical[self.normalize(canonical)] = canonical
        for alias in entry["aliases"]:
            self._alias_to_canonical[self.normalize(alias)] = canonical
        return entry

    def _prime(self) -> None:
        if self._primed:
            return
        self._primed = True
        try:
            rows = self.client.run_read(
                "MATCH (k:Concept) RETURN k.name AS name, k.type AS type, "
                "k.aliases AS aliases LIMIT $limit",
                {"limit": self.max_existing},
            )
        except Exception:
            rows = []
        for row in rows or []:
            canonical = row.get("name")
            if not canonical:
                continue
            aliases = {a for a in (row.get("aliases") or []) if a}
            vec = self._embed(canonical)
            self._register(canonical, row.get("type") or DEFAULT_NODE_TYPE, aliases, vec)

    def resolve(
        self,
        name: str,
        description: str = "",
        node_type: str = DEFAULT_NODE_TYPE,
    ) -> str:
        """Returneaza numele canonic pentru o forma de suprafata (creeaza unul nou la nevoie)."""
        norm = self.normalize(name)
        if not norm:
            return name
        self._prime()

        existing = self._alias_to_canonical.get(norm)
        if existing:
            return existing

        if self.embedder is not None:
            query_vec = self._embed(f"{name}. {description}".strip(". "))
            if query_vec is not None:
                best_entry, best_sim = None, 0.0
                for entry in self._index:
                    if entry["vec"] is None:
                        continue
                    sim = _cosine(query_vec, entry["vec"])
                    if sim > best_sim:
                        best_entry, best_sim = entry, sim
                if best_entry is not None and best_sim >= self.threshold:
                    best_entry["aliases"].add(norm)
                    self._alias_to_canonical[norm] = best_entry["canonical"]
                    return best_entry["canonical"]
                # Concept nou — il inregistram cu embedding pentru potriviri ulterioare.
                self._register(norm, node_type, set(), query_vec)
                return norm

        self._register(norm, node_type, set())
        return norm


class GraphIngestionService:
    """Offline incremental graph ingestion service with confidence gating."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        min_confidence: float = 0.70,
        extraction_mode: str = "heuristic_fallback",
        structured_extractor: Optional[Callable[[str], Dict[str, Any]]] = None,
        embedder: Optional[Callable[[str], Optional[List[float]]]] = None,
        resolver_threshold: float = 0.86,
        batch_extractor: Optional[Callable[[List[str]], List[Dict[str, Any]]]] = None,
        extraction_batch_size: int = 5,
    ):
        self.client = neo4j_client
        self.min_confidence = float(min_confidence)
        self.extraction_mode = (extraction_mode or "heuristic_fallback").strip().lower()
        self.structured_extractor = structured_extractor
        # Extractie pe loturi (Phase 4 / free tier): 1 apel LLM pentru N chunk-uri.
        self.batch_extractor = batch_extractor
        self.extraction_batch_size = max(1, int(extraction_batch_size))
        # Entity resolution: deduplica concepte la scriere (Phase 2).
        self.resolver = ConceptResolver(
            neo4j_client, embedder=embedder, threshold=resolver_threshold
        )

    def _extract_payloads_for_contents(
        self,
        contents: List[str],
    ) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """Extrage (concepte, relatii) pentru o lista de texte, aliniat la input.

        Cand batch_extractor e disponibil (mod llm_structured), trimite cate
        `extraction_batch_size` texte intr-un singur apel LLM -> mult mai putine
        request-uri (esential pe free tier). Altfel cade pe extractie per-text.
        """
        if not contents:
            return []

        use_batch = (
            self.extraction_mode == "llm_structured"
            and self.batch_extractor is not None
        )
        if not use_batch:
            return [self._extract_graph_payload(text) for text in contents]

        results: List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = []
        for start in range(0, len(contents), self.extraction_batch_size):
            batch = contents[start : start + self.extraction_batch_size]
            try:
                raw_list = self.batch_extractor(batch) or []
            except Exception as exc:
                logger.warning("Batch graph extraction failed, fallback per-text: %s", exc)
                raw_list = []
            for idx in range(len(batch)):
                raw = raw_list[idx] if idx < len(raw_list) and isinstance(raw_list[idx], dict) else {}
                # FARA fallback heuristic in llm_structured: un lot esuat (429/503) sau
                # gol nu mai polueaza graful cu zgomot. Chunk-urile raman ne-extrase si
                # sunt completate la un re-run (cache-ul sare peste cele deja facute).
                results.append(self._normalize_payload(raw))
        return results

    @property
    def enabled(self) -> bool:
        return self.client.enabled

    def ingest_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Ingest chunks + concept/relation structure, INCREMENTAL lot cu lot.

        Pentru fiecare lot: 1 apel LLM -> scriere imediata in graf. Daca rebuild-ul
        e intrerupt (cota/overload), loturile deja procesate raman scrise; un re-run
        continua (cache-ul de extractie sare peste chunk-urile deja facute).
        """
        if not self.enabled:
            return {"documents": 0, "chunks": 0, "concepts": 0, "relations": 0}

        valid_chunks = [
            chunk for chunk in chunks
            if (chunk.get("metadata") or {}).get("source_path")
        ]

        unique_docs: set = set()
        chunks_count = 0
        concepts_count = 0
        relations_count = 0

        batch_size = self.extraction_batch_size
        for start in range(0, len(valid_chunks), batch_size):
            batch_chunks = valid_chunks[start : start + batch_size]
            payloads = self._extract_payloads_for_contents(
                [chunk.get("content", "") for chunk in batch_chunks]
            )
            for chunk, (concepts, relations) in zip(batch_chunks, payloads):
                c_inc, r_inc = self._ingest_one_chunk(chunk, concepts, relations, unique_docs)
                chunks_count += 1
                concepts_count += c_inc
                relations_count += r_inc

        return {
            "documents": len(unique_docs),
            "chunks": chunks_count,
            "concepts": concepts_count,
            "relations": relations_count,
        }

    def _ingest_one_chunk(
        self,
        chunk: Dict[str, Any],
        concepts: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        unique_docs: set,
    ) -> Tuple[int, int]:
        """Scrie un singur chunk: scaffolding + concepte + relatii. Returneaza (concepte, relatii) scrise."""
        metadata = chunk.get("metadata", {})
        content = chunk.get("content", "")
        source_path = metadata.get("source_path", "")
        doc_id = str(metadata.get("doc_id") or source_path)
        filename = metadata.get("filename", "document")
        collection = metadata.get("collection", "general")
        chunk_id = f"{doc_id}:{metadata.get('chunk_id', 0)}"
        chunk_order = int(metadata.get("chunk_id", 0))
        topic_id = f"topic:{(collection or 'general').lower()}"
        unique_docs.add(doc_id)

        self.client.run_write(
            """
            MERGE (t:Topic {id: $topic_id})
            SET t.name = $topic_name
            MERGE (d:Document {id: $doc_id})
            SET d.filename = $filename,
                d.collection = $collection,
                d.source_path = $source_path
            MERGE (d)-[:ABOUT_TOPIC]->(t)
            MERGE (c:Chunk {id: $chunk_id})
            SET c.order = $chunk_order,
                c.text = $chunk_text
            MERGE (c)-[:SOURCED_FROM]->(d)
            """,
            {
                "topic_id": topic_id,
                "topic_name": collection,
                "doc_id": doc_id,
                "filename": filename,
                "collection": collection,
                "source_path": source_path,
                "chunk_id": str(chunk_id),
                "chunk_order": chunk_order,
                "chunk_text": content[:3000],
            },
        )

        concepts_written = 0
        relations_written = 0
        type_by_name: Dict[str, str] = {}
        for concept in concepts:
            if not self._concept_confidence_ok(concept):
                continue
            canonical = self._write_concept_mention(
                concept,
                topic_id=topic_id,
                anchor_label="Chunk",
                anchor_id=chunk_id,
            )
            type_by_name[canonical] = concept.get("type", DEFAULT_NODE_TYPE)
            concepts_written += 1

        for relation in relations:
            if self._write_relation(
                relation,
                topic=collection,
                type_by_name=type_by_name,
                chunk_id=chunk_id,
                document_id=doc_id,
            ):
                relations_written += 1

        return concepts_written, relations_written

    def _extract_graph_payload(
        self,
        text: str,
        max_concepts: int = 10,
        max_relations: int = 12,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract concepts and relations using configured mode:
        - llm_structured: strict JSON extraction + heuristic fallback
        - heuristic_fallback: heuristic extraction only

        Both modes return the same dict shapes:
        concept: {name, confidence, type, aliases, description}
        relation: {source, target, type, confidence, evidence}
        """
        if (
            self.extraction_mode == "llm_structured"
            and self.structured_extractor is not None
        ):
            llm_concepts, llm_relations = self._extract_structured_graph_llm(
                text=text,
                max_concepts=max_concepts,
                max_relations=max_relations,
            )
            if llm_concepts or llm_relations:
                return llm_concepts, llm_relations

        return self._extract_structured_graph(
            text=text,
            max_concepts=max_concepts,
            max_relations=max_relations,
        )

    # ------------------------------------------------------------------
    # Write helpers (partajate intre ingestia de chunks si memory artifacts)
    # ------------------------------------------------------------------

    def _concept_confidence_ok(self, concept: Dict[str, Any]) -> bool:
        return float(concept.get("confidence", 0.0)) >= self.min_confidence

    def _write_concept_mention(
        self,
        concept: Dict[str, Any],
        *,
        topic_id: str,
        anchor_label: str,
        anchor_id: str,
    ) -> None:
        """Upsert un Concept (cu type + description) si muchia MENTIONS de la sursa.

        anchor_label e 'Chunk' sau 'Artifact' (valoare interna, fara input user).
        """
        name = self._resolve_concept_name(concept)
        # Formele de suprafata (numele extras + alias-uri) se acumuleaza ca aliases.
        surfaces = [concept.get("name", name)] + list(concept.get("aliases", []) or [])
        aliases: List[str] = []
        for surface in surfaces:
            norm = ConceptResolver.normalize(surface)
            if norm and norm != name and norm not in aliases:
                aliases.append(norm)

        cypher = f"""
            MERGE (k:Concept {{name: $name}})
            SET k.type = coalesce(k.type, $type),
                k.description = CASE
                    WHEN k.description IS NULL OR k.description = '' THEN $description
                    ELSE k.description END,
                k.aliases = coalesce(k.aliases, []) +
                    [a IN $aliases WHERE NOT a IN coalesce(k.aliases, [])]
            MERGE (src:{anchor_label} {{id: $anchor_id}})
            MERGE (t:Topic {{id: $topic_id}})
            MERGE (src)-[r:MENTIONS]->(k)
            SET r.confidence = $confidence
            MERGE (k)-[:RELATED_TO]->(t)
        """
        self.client.run_write(
            cypher,
            {
                "name": name,
                "type": concept.get("type", DEFAULT_NODE_TYPE),
                "description": concept.get("description", ""),
                "aliases": aliases,
                "anchor_id": str(anchor_id),
                "topic_id": topic_id,
                "confidence": float(concept.get("confidence", 0.0)),
            },
        )
        return name

    @staticmethod
    def _relation_schema_ok(
        relation_type: str,
        source_type: Optional[str],
        target_type: Optional[str],
    ) -> bool:
        """Valideaza (tip relatie, tip sursa, tip tinta) fata de ontologie.

        Lenient cand tipul unui capat e necunoscut (None) — nu respingem din lipsa de tip.
        """
        schema = RELATION_SCHEMA.get(relation_type)
        if schema is None:
            return False
        src_allowed, tgt_allowed = schema
        if src_allowed and source_type and source_type not in src_allowed:
            return False
        if tgt_allowed and target_type and target_type not in tgt_allowed:
            return False
        return True

    def _write_relation(
        self,
        relation: Dict[str, Any],
        *,
        topic: str,
        type_by_name: Optional[Dict[str, str]] = None,
        chunk_id: str = "",
        document_id: str = "",
        artifact_id: str = "",
    ) -> bool:
        """Upsert o muchie tipata intre concepte, cu evidence + corroborare + schema.

        Returneaza True daca muchia a fost scrisa (trecuta de gating + schema).
        """
        relation_type = relation.get("type", "RELATED_TO")
        confidence = float(relation.get("confidence", 0.0))
        if relation_type not in ALLOWED_RELATIONS or confidence < self.min_confidence:
            return False

        source = self._resolve_existing_name(relation["source"])
        target = self._resolve_existing_name(relation["target"])
        if not source or not target or source == target:
            return False

        types = type_by_name or {}
        if not self._relation_schema_ok(relation_type, types.get(source), types.get(target)):
            return False

        # relation_type vine dintr-un whitelist (ALLOWED_RELATIONS) -> sigur in f-string.
        # Confidence corroborata: base = max peste observatii; +0.05 / observatie suplimentara.
        cypher = f"""
            MERGE (a:Concept {{name: $source}})
            MERGE (b:Concept {{name: $target}})
            MERGE (a)-[r:{relation_type}]->(b)
            SET r.base_confidence = CASE
                    WHEN coalesce(r.base_confidence, 0.0) > $confidence THEN r.base_confidence
                    ELSE $confidence END,
                r.observations = coalesce(r.observations, 0) + 1,
                r.evidence = $evidence,
                r.chunk_id = $chunk_id,
                r.document_id = $document_id,
                r.artifact_id = $artifact_id,
                r.topic = $topic
            SET r.confidence = CASE
                    WHEN r.base_confidence + 0.05 * (r.observations - 1) > 1.0 THEN 1.0
                    ELSE r.base_confidence + 0.05 * (r.observations - 1) END
        """
        self.client.run_write(
            cypher,
            {
                "source": source,
                "target": target,
                "confidence": confidence,
                "evidence": relation.get("evidence", ""),
                "chunk_id": str(chunk_id or ""),
                "document_id": str(document_id or ""),
                "artifact_id": str(artifact_id or ""),
                "topic": topic,
            },
        )
        return True

    def _resolve_concept_name(self, concept: Dict[str, Any]) -> str:
        """Rezolva forma de suprafata la conceptul canonic (entity resolution)."""
        return self.resolver.resolve(
            concept["name"],
            description=concept.get("description", ""),
            node_type=concept.get("type", DEFAULT_NODE_TYPE),
        )

    def _resolve_existing_name(self, name: str) -> str:
        """Rezolva un capat de relatie la conceptul canonic existent (sau nou)."""
        return self.resolver.resolve(name)

    def _extract_structured_graph_llm(
        self,
        text: str,
        max_concepts: int = 10,
        max_relations: int = 12,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if self.structured_extractor is None:
            return [], []
        try:
            payload = self.structured_extractor(text) or {}
        except Exception as exc:
            logger.warning("Structured graph extraction failed, fallback heuristic: %s", exc)
            return [], []
        return self._normalize_payload(payload, max_concepts, max_relations)

    def _normalize_payload(
        self,
        payload: Dict[str, Any],
        max_concepts: int = 10,
        max_relations: int = 12,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Normalizeaza un payload brut de extractie (LLM) la forma canonica de dict-uri."""
        if not isinstance(payload, dict):
            return [], []
        concepts_raw = payload.get("concepts", []) or []
        relations_raw = payload.get("relations", []) or []

        concepts: List[Dict[str, Any]] = []
        seen_concepts = set()
        for item in concepts_raw:
            if isinstance(item, dict):
                concept_name = self._canonical(str(item.get("name", "")))
                confidence = float(item.get("confidence", 0.0) or 0.0)
                node_type = str(item.get("type", "") or DEFAULT_NODE_TYPE).strip()
                if node_type not in NODE_TYPES:
                    node_type = DEFAULT_NODE_TYPE
                aliases = [
                    self._canonical(str(a))
                    for a in (item.get("aliases") or [])
                    if str(a).strip()
                ]
                aliases = [a for a in aliases if a and a != concept_name]
                description = str(item.get("description", "") or "").strip()[:500]
            else:
                concept_name = self._canonical(str(item))
                confidence = 0.75
                node_type = DEFAULT_NODE_TYPE
                aliases = []
                description = ""
            if not concept_name or concept_name in seen_concepts:
                continue
            seen_concepts.add(concept_name)
            concepts.append(
                {
                    "name": concept_name,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "type": node_type,
                    "aliases": aliases,
                    "description": description,
                }
            )
            if len(concepts) >= max_concepts:
                break

        relations: List[Dict[str, Any]] = []
        seen_relations = set()
        for item in relations_raw:
            if not isinstance(item, dict):
                continue
            source = self._canonical(str(item.get("source", "")))
            target = self._canonical(str(item.get("target", "")))
            relation_type = str(item.get("type", "RELATED_TO")).strip().upper()
            confidence = float(item.get("confidence", 0.0) or 0.0)
            evidence = str(item.get("evidence", "") or "").strip()[:300]

            if not source or not target or source == target:
                continue
            if relation_type not in ALLOWED_RELATIONS:
                relation_type = "RELATED_TO"

            relation_key = (source, target, relation_type)
            if relation_key in seen_relations:
                continue
            seen_relations.add(relation_key)

            relations.append(
                {
                    "source": source,
                    "target": target,
                    "type": relation_type,
                    "confidence": max(0.0, min(1.0, confidence)),
                    "evidence": evidence,
                }
            )
            if len(relations) >= max_relations:
                break

        return concepts, relations

    def ingest_memory_artifacts(self, artifacts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Project canonical Postgres memory artifacts into Neo4j."""
        if not self.enabled:
            return {"artifacts": 0, "concepts": 0, "relations": 0}

        valid_artifacts = [a for a in artifacts if str(a.get("id") or "")]
        payloads = self._extract_payloads_for_contents(
            [(a.get("content", "") or a.get("title", "")) for a in valid_artifacts]
        )

        artifacts_count = 0
        concepts_count = 0
        relations_count = 0

        for artifact, (concepts, relations) in zip(valid_artifacts, payloads):
            artifact_id = str(artifact.get("id") or "")
            artifact_type = artifact.get("artifact_type", "memory")
            title = artifact.get("title", "")
            content = artifact.get("content", "")
            collection = artifact.get("topic_collection", "") or "general"
            topic_id = f"topic:{collection.lower()}"

            self.client.run_write(
                """
                MERGE (t:Topic {id: $topic_id})
                SET t.name = $topic_name
                MERGE (a:Artifact {id: $artifact_id})
                SET a.type = $artifact_type,
                    a.title = $title,
                    a.source_table = $source_table,
                    a.source_id = $source_id,
                    a.content = $content
                MERGE (a)-[:ABOUT_TOPIC]->(t)
                """,
                {
                    "topic_id": topic_id,
                    "topic_name": collection,
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_type,
                    "title": title,
                    "source_table": artifact.get("source_table", ""),
                    "source_id": str(artifact.get("source_id", "")),
                    "content": content[:3000],
                },
            )
            artifacts_count += 1

            type_by_name: Dict[str, str] = {}
            for concept in concepts:
                if not self._concept_confidence_ok(concept):
                    continue
                canonical = self._write_concept_mention(
                    concept,
                    topic_id=topic_id,
                    anchor_label="Artifact",
                    anchor_id=artifact_id,
                )
                type_by_name[canonical] = concept.get("type", DEFAULT_NODE_TYPE)
                concepts_count += 1

            for relation in relations:
                if self._write_relation(
                    relation,
                    topic=collection,
                    type_by_name=type_by_name,
                    artifact_id=artifact_id,
                ):
                    relations_count += 1

        return {
            "artifacts": artifacts_count,
            "concepts": concepts_count,
            "relations": relations_count,
        }

    def rebuild_projection(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        memory_artifacts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, int]:
        """Delete derived Neo4j projection and rebuild it from Postgres data."""
        if not self.enabled:
            return {
                "deleted": 0,
                "documents": 0,
                "chunks": 0,
                "artifacts": 0,
                "concepts": 0,
                "relations": 0,
            }

        # Numara intai, apoi sterge. (Nu combina intr-un singur MATCH...WITH
        # count(n)...DETACH DELETE n: dupa WITH, variabila `n` iese din scope.)
        count_rows = self.client.run_read("MATCH (n) RETURN count(n) AS count", {})
        deleted = int(count_rows[0].get("count", 0)) if count_rows else 0
        self.client.run_write("MATCH (n) DETACH DELETE n", {})
        # Graf gol -> reseteaza indexul de entity resolution (fara concepte fantoma).
        self.resolver.reset()

        chunk_result = self.ingest_chunks(chunks or [])
        artifact_result = self.ingest_memory_artifacts(memory_artifacts or [])
        return {
            "deleted": deleted,
            "documents": int(chunk_result.get("documents", 0)),
            "chunks": int(chunk_result.get("chunks", 0)),
            "artifacts": int(artifact_result.get("artifacts", 0)),
            "concepts": int(chunk_result.get("concepts", 0)) + int(artifact_result.get("concepts", 0)),
            "relations": int(chunk_result.get("relations", 0)) + int(artifact_result.get("relations", 0)),
        }

    def ingest_decision(
        self,
        decision_id: str,
        title: str,
        rationale: str,
        topic_collection: str,
    ) -> bool:
        if not self.enabled:
            return False

        return self.client.run_write(
            """
            MERGE (t:Topic {id: $topic_id})
            SET t.name = $topic_name
            MERGE (d:Decision {id: $decision_id})
            SET d.title = $title,
                d.rationale = $rationale
            MERGE (d)-[:DECIDED_IN]->(t)
            """,
            {
                "topic_id": f"topic:{(topic_collection or 'general').lower()}",
                "topic_name": topic_collection or "general",
                "decision_id": decision_id,
                "title": title,
                "rationale": rationale[:4000],
            },
        )

    # ------------------------------------------------------------------
    # Governance: revizuire / curatare graf (Phase 3)
    # ------------------------------------------------------------------

    def list_low_confidence_relations(
        self, threshold: float = 0.78, limit: int = 30
    ) -> List[Dict[str, Any]]:
        """Muchii concept-concept cu confidence sub prag, pentru revizuire manuala."""
        if not self.enabled:
            return []
        rows = self.client.run_read(
            """
            MATCH (a:Concept)-[r]->(b:Concept)
            WHERE coalesce(r.confidence, 0.0) < $threshold
            RETURN a.name AS source, b.name AS target, type(r) AS type,
                   coalesce(r.confidence, 0.0) AS confidence,
                   coalesce(r.observations, 1) AS observations,
                   coalesce(r.evidence, '') AS evidence
            ORDER BY confidence ASC
            LIMIT $limit
            """,
            {"threshold": float(threshold), "limit": int(limit)},
        )
        return [
            {
                "source": row.get("source", ""),
                "target": row.get("target", ""),
                "type": row.get("type", "RELATED_TO"),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "observations": int(row.get("observations", 1) or 1),
                "evidence": row.get("evidence", "") or "",
            }
            for row in (rows or [])
            if row.get("source") and row.get("target")
        ]

    def delete_relation(self, source: str, target: str, relation_type: str) -> bool:
        """Sterge o muchie tipata gresita intre doua concepte."""
        if not self.enabled or relation_type not in ALLOWED_RELATIONS:
            return False
        return self.client.run_write(
            f"""
            MATCH (a:Concept {{name: $source}})-[r:{relation_type}]->(b:Concept {{name: $target}})
            DELETE r
            """,
            {"source": source, "target": target},
        )

    def list_concepts(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Concepte cu tip, aliasuri si numar de mentiuni (pentru revizuire/merge)."""
        if not self.enabled:
            return []
        rows = self.client.run_read(
            """
            MATCH (k:Concept)
            OPTIONAL MATCH (ch:Chunk)-[:MENTIONS]->(k)
            RETURN k.name AS name, coalesce(k.type, 'Concept') AS type,
                   coalesce(k.aliases, []) AS aliases, count(ch) AS mentions
            ORDER BY mentions DESC
            LIMIT $limit
            """,
            {"limit": int(limit)},
        )
        return [
            {
                "name": row.get("name", ""),
                "type": row.get("type", "Concept"),
                "aliases": [a for a in (row.get("aliases") or []) if a],
                "mentions": int(row.get("mentions", 0) or 0),
            }
            for row in (rows or [])
            if row.get("name")
        ]

    def merge_concepts(self, from_name: str, into_name: str) -> bool:
        """Uneste conceptul `from_name` in `into_name`: redirecteaza muchiile, sterge sursa.

        Implementare fara APOC (per tip de relatie din ontologie), best-effort.
        """
        if not self.enabled:
            return False
        from_c = self.resolver.normalize(from_name)
        into_c = self.resolver.normalize(into_name)
        if not from_c or not into_c or from_c == into_c:
            return False
        try:
            # 1. Mosteneste numele + aliasurile sursei ca aliasuri pe tinta.
            self.client.run_write(
                """
                MATCH (f:Concept {name: $from}), (i:Concept {name: $into})
                SET i.aliases = coalesce(i.aliases, []) +
                    [a IN ([f.name] + coalesce(f.aliases, []))
                     WHERE NOT a IN coalesce(i.aliases, []) AND a <> i.name]
                """,
                {"from": from_c, "into": into_c},
            )
            # 2. Redirecteaza MENTIONS si RELATED_TO->Topic.
            self.client.run_write(
                """
                MATCH (src)-[:MENTIONS]->(:Concept {name: $from})
                MATCH (i:Concept {name: $into})
                MERGE (src)-[:MENTIONS]->(i)
                """,
                {"from": from_c, "into": into_c},
            )
            self.client.run_write(
                """
                MATCH (:Concept {name: $from})-[:RELATED_TO]->(t:Topic)
                MATCH (i:Concept {name: $into})
                MERGE (i)-[:RELATED_TO]->(t)
                """,
                {"from": from_c, "into": into_c},
            )
            # 3. Redirecteaza muchiile concept-concept (ambele directii), per tip.
            for rtype in ALLOWED_RELATIONS:
                self.client.run_write(
                    f"""
                    MATCH (:Concept {{name: $from}})-[r:{rtype}]->(b:Concept)
                    WHERE b.name <> $into
                    MATCH (i:Concept {{name: $into}})
                    MERGE (i)-[nr:{rtype}]->(b)
                    SET nr.confidence = coalesce(nr.confidence, r.confidence),
                        nr.evidence = coalesce(nr.evidence, r.evidence)
                    """,
                    {"from": from_c, "into": into_c},
                )
                self.client.run_write(
                    f"""
                    MATCH (a:Concept)-[r:{rtype}]->(:Concept {{name: $from}})
                    WHERE a.name <> $into
                    MATCH (i:Concept {{name: $into}})
                    MERGE (a)-[nr:{rtype}]->(i)
                    SET nr.confidence = coalesce(nr.confidence, r.confidence),
                        nr.evidence = coalesce(nr.evidence, r.evidence)
                    """,
                    {"from": from_c, "into": into_c},
                )
            # 4. Sterge conceptul sursa.
            self.client.run_write(
                "MATCH (f:Concept {name: $from}) DETACH DELETE f",
                {"from": from_c},
            )
            return True
        except Exception as exc:
            logger.error("merge_concepts failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Community detection + summaries (Phase 4)
    # ------------------------------------------------------------------

    def rebuild_communities(
        self,
        summarizer: Optional[Callable[[List[str], List[str]], str]] = None,
        max_communities: int = 12,
        min_size: int = 3,
    ) -> Dict[str, Any]:
        """Detecteaza comunitati de concepte (componente conexe) + rezumate LLM.

        Componente conexe = abordare fara GDS/APOC (merge pe Aura). Comunitatile
        prea mici sunt ignorate; cele mari sunt rezumate si stocate ca noduri
        :Community cu muchii (:Concept)-[:IN_COMMUNITY]->(:Community).
        """
        if not self.enabled:
            return {"communities": 0, "summarized": 0}

        edges = self.client.run_read(
            "MATCH (a:Concept)-[r]->(b:Concept) RETURN a.name AS source, b.name AS target",
            {},
        ) or []

        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for edge in edges:
            src, tgt = edge.get("source"), edge.get("target")
            if src and tgt:
                union(src, tgt)

        groups: Dict[str, List[str]] = {}
        for node in list(parent.keys()):
            groups.setdefault(find(node), []).append(node)

        communities = sorted(
            (members for members in groups.values() if len(members) >= min_size),
            key=len,
            reverse=True,
        )[:max_communities]

        # Reseteaza proiectia de comunitati.
        self.client.run_write("MATCH (com:Community) DETACH DELETE com", {})

        summarized = 0
        for idx, members in enumerate(communities):
            desc_rows = self.client.run_read(
                "MATCH (k:Concept) WHERE k.name IN $members "
                "RETURN coalesce(k.description, '') AS d",
                {"members": members},
            ) or []
            descriptions = [r.get("d", "") for r in desc_rows]

            summary = ""
            if summarizer is not None:
                try:
                    summary = summarizer(members, descriptions) or ""
                except Exception as exc:
                    logger.warning("community summarizer failed: %s", exc)
                    summary = ""
            if not summary:
                summary = ", ".join(members[:3])

            self.client.run_write(
                """
                MERGE (com:Community {id: $cid})
                SET com.summary = $summary, com.size = $size
                WITH com
                UNWIND $members AS m
                MATCH (k:Concept {name: m})
                MERGE (k)-[:IN_COMMUNITY]->(com)
                """,
                {
                    "cid": f"community:{idx}",
                    "summary": summary,
                    "size": len(members),
                    "members": members,
                },
            )
            summarized += 1

        return {"communities": len(communities), "summarized": summarized}

    def _extract_structured_graph(
        self,
        text: str,
        max_concepts: int = 10,
        max_relations: int = 12,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Heuristic fallback: lexical concepts + sentence co-occurrence relations.

        Pastrat doar ca plasa de siguranta cand extractia LLM nu e disponibila.
        Emite aceeasi forma de dict ca extractia structurata (type/aliases/desc).
        """
        concept_names = self._extract_concepts(text, max_concepts=max_concepts)
        concept_dicts: List[Dict[str, Any]] = [
            {
                "name": name,
                "confidence": 0.78,
                "type": DEFAULT_NODE_TYPE,
                "aliases": [],
                "description": "",
            }
            for name in concept_names
        ]
        if len(concept_names) < 2:
            return concept_dicts, []

        relations: List[Dict[str, Any]] = []
        seen = set()
        sentences = re.split(r"[.!?\n]+", text)
        for sentence in sentences:
            sentence_clean = sentence.strip().lower()
            if not sentence_clean:
                continue

            sentence_concepts = self._extract_concepts(sentence, max_concepts=4)
            if len(sentence_concepts) < 2:
                continue

            relation_type = self._detect_relation_type(sentence_clean)
            base_confidence = 0.82 if relation_type != "RELATED_TO" else 0.72
            evidence = sentence.strip()[:300]

            for left, right in combinations(sentence_concepts[:4], 2):
                left_norm = self._canonical(left)
                right_norm = self._canonical(right)
                if not left_norm or not right_norm or left_norm == right_norm:
                    continue
                key = (left_norm, right_norm, relation_type)
                if key in seen:
                    continue
                seen.add(key)
                relations.append(
                    {
                        "source": left_norm,
                        "target": right_norm,
                        "type": relation_type,
                        "confidence": base_confidence,
                        "evidence": evidence,
                    }
                )
                if len(relations) >= max_relations:
                    return concept_dicts, relations

        return concept_dicts, relations

    def _detect_relation_type(self, sentence: str) -> str:
        for relation_type, markers in RELATION_MARKERS.items():
            if any(marker in sentence for marker in markers):
                return relation_type
        return "RELATED_TO"

    @classmethod
    def _extract_concepts(cls, text: str, max_concepts: int = 8) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_\-]{4,}", text.lower())
        filtered = [
            cls._canonical(token)
            for token in tokens
            if token not in STOPWORDS and not token.isdigit()
        ]
        filtered = [token for token in filtered if token]
        if not filtered:
            return []

        concepts: List[str] = []
        seen = set()
        for token in filtered:
            if token in seen:
                continue
            seen.add(token)
            concepts.append(token)
            if len(concepts) >= max_concepts:
                break
        return concepts

    @staticmethod
    def _canonical(value: str) -> str:
        cleaned = re.sub(r"\s+", " ", (value or "").strip().lower())
        cleaned = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", cleaned)
        return cleaned
