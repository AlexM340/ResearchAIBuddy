"""
Modulul RAG principal pentru CerebrumAI (Asistentul Personalizat de Cercetare și Învățare)
Implementează sistemul RAG avansat cu Gemini 2.5 Flash
"""

import os
import json
import time
import hashlib
import re
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import requests

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Google GenAI SDK nu este disponibil")

try:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    from langchain_community.document_loaders import PyPDFLoader

    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain nu este disponibil")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Sentence Transformers nu este disponibil")

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

try:
    from diskcache import Cache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False

try:
    from storage import PostgresClient, SecondBrainRepository
    STORAGE_AVAILABLE = True
except Exception:
    PostgresClient = None
    SecondBrainRepository = None
    STORAGE_AVAILABLE = False

try:
    from graph import Neo4jClient, GraphIngestionService, GraphQueryService
    GRAPH_AVAILABLE = True
except Exception:
    Neo4jClient = None
    GraphIngestionService = None
    GraphQueryService = None
    GRAPH_AVAILABLE = False

try:
    from query import (
        QueryIntent,
        RetrievalPlan,
        EvidenceItem,
        DecisionMemory,
        RuleBasedIntentRouter,
        HybridEvidenceFusion,
        ContextBuilder,
    )
    QUERY_INTEL_AVAILABLE = True
except Exception:
    QueryIntent = None
    RetrievalPlan = None
    EvidenceItem = None
    DecisionMemory = None
    RuleBasedIntentRouter = None
    HybridEvidenceFusion = None
    ContextBuilder = None
    QUERY_INTEL_AVAILABLE = False

# Configurare logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configurare pentru sistemul RAG"""
    model_name: str = "gemini-2.5-flash"
    fallback_model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.1
    max_tokens: int = 65536  # Increased to 128K for better performance
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_context_docs: int = 5
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    cache_enabled: bool = True
    rate_limit_rpm: int = 30
    rate_limit_rpd: int = 3000
    retrieval_candidates: int = 20
    rerank_top_k: int = 8
    neighbor_window: int = 1
    hybrid_alpha: float = 0.7
    # Second Brain configuration
    retrieval_mode: str = "auto"
    vector_top_k: int = 8
    graph_top_k_paths: int = 6
    hybrid_rerank_top_k: int = 8
    vector_confidence_threshold: float = 0.35
    context_budget_tokens: int = 8000
    vector_budget_ratio: float = 0.65
    graph_budget_ratio: float = 0.35
    enable_graph_rag: bool = True
    router_mode: str = "rule_based"
    # Retrieval quality (P3): reranker + HyDE
    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_input_k: int = 20  # cat oversampling fata de top_k final
    enable_hyde: bool = False  # genereaza pseudo-document pt expansion (extra LLM call)
    hyde_max_tokens: int = 200
    # Web search grounded (P4): Tavily
    tavily_api_key: str = ""
    enable_web_fallback: bool = True  # auto fallback cand context intern insuficient
    web_fallback_max_results: int = 5
    web_fallback_search_depth: str = "advanced"  # basic | advanced
    decision_extraction_enabled: bool = True
    graph_max_hops: int = 2
    graph_min_confidence: float = 0.70
    graph_extraction_mode: str = "heuristic_fallback"
    memory_consolidation_enabled: bool = True
    memory_decay_days: int = 90
    db_primary_strict: bool = True
    ingestion_embedding_batch_size: int = 64
    ingestion_max_workers: int = 4
    postgres_dsn: str = ""
    pgvector_embedding_dim: int = 384
    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""

class GeminiRateLimiter:
    """Rate limiter optimizat pentru Gemini Flash"""
    
    def __init__(self, requests_per_minute: int = 30, requests_per_day: int = 3000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.minute_requests = []
        self.day_requests = []
        self.total_requests = 0
        
    def can_make_request(self) -> bool:
        """Verifică dacă se poate face o cerere"""
        current_time = time.time()
        
        # Curață cererile mai vechi de 1 minut
        self.minute_requests = [t for t in self.minute_requests if current_time - t < 60]
        
        # Curață cererile mai vechi de 1 zi
        self.day_requests = [t for t in self.day_requests if current_time - t < 86400]
        
        return (len(self.minute_requests) < self.requests_per_minute and 
                len(self.day_requests) < self.requests_per_day)
    
    def add_request(self):
        """Înregistrează o cerere"""
        current_time = time.time()
        self.minute_requests.append(current_time)
        self.day_requests.append(current_time)
        self.total_requests += 1
    
    def wait_if_needed(self):
        """Așteaptă dacă este necesar"""
        if not self.can_make_request():
            wait_time = 60 / self.requests_per_minute
            logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    
    def get_status(self) -> Dict[str, Any]:
        """Returnează statusul rate limiter-ului"""
        return {
            "total_requests": self.total_requests,
            "minute_requests": len(self.minute_requests),
            "day_requests": len(self.day_requests),
            "can_make_request": self.can_make_request()
        }

class SimpleCache:
    """Cache simplu bazat pe dicționar pentru răspunsuri"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.stats = {"hits": 0, "misses": 0, "total_saved_tokens": 0}
    
    def _generate_key(self, prompt: str, model_settings: Dict[str, Any]) -> str:
        """Generează o cheie unică pentru prompt și setări"""
        content = f"{prompt}_{json.dumps(model_settings, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_cached_response(self, prompt: str, model_settings: Dict[str, Any]) -> Optional[str]:
        """Obține un răspuns din cache dacă există"""
        key = self._generate_key(prompt, model_settings)
        
        if key in self.cache:
            self.stats["hits"] += 1
            self.stats["total_saved_tokens"] += len(self.cache[key]["response"])
            logger.info(f"Cache hit pentru prompt: {prompt[:50]}...")
            return self.cache[key]["response"]
        
        self.stats["misses"] += 1
        return None
    
    def cache_response(self, prompt: str, response: str, model_settings: Dict[str, Any]):
        """Salvează un răspuns în cache"""
        if len(response) < 50:  # Nu cache răspunsuri foarte scurte
            return
            
        key = self._generate_key(prompt, model_settings)
        
        # Curată cache-ul dacă este prea mare
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "model_settings": model_settings
        }
        logger.debug(f"Răspuns salvat în cache pentru: {prompt[:50]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Returnează statisticile cache-ului"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / max(total_requests, 1)) * 100
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate,
            "cache_size": len(self.cache)
        }
    
    def clear_cache(self):
        """Curată cache-ul"""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0, "total_saved_tokens": 0}

class PersistentEmbeddingsCache:
    """Cache persistent pentru embeddings cu verificare hash MD5 si tracking model."""

    def __init__(self, cache_dir: str = "./data/embeddings_cache", model_name: str = ""):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Fișiere pentru cache
        self.metadata_file = self.cache_dir / "metadata.json"
        self.embeddings_file = self.cache_dir / "embeddings.pkl"
        self.documents_file = self.cache_dir / "documents.pkl"
        self.model_signature_file = self.cache_dir / "model_signature.json"

        # Încarcă cache-ul existent
        self.metadata = self._load_metadata()
        self.embeddings_cache = {}
        self.documents_cache = {}

        self.model_name = (model_name or "").strip()
        self._enforce_model_signature()

        logger.info(f"Cache persistent inițializat în: {self.cache_dir}")

    def _enforce_model_signature(self) -> None:
        """Daca modelul s-a schimbat fata de cache, invalideaza cache-ul si marcheaza reindex pending."""
        if not self.model_name:
            return
        sig = self._read_signature()
        saved_model = sig.get("model_name", "")
        pending = bool(sig.get("reindex_pending", False))
        if saved_model and saved_model != self.model_name:
            logger.warning(
                "Embedding model change detected (was %s, now %s). Invalidating local cache.",
                saved_model, self.model_name,
            )
            self.clear_cache()
            pending = True
        self._write_signature(self.model_name, pending)

    def _read_signature(self) -> Dict[str, Any]:
        if not self.model_signature_file.exists():
            return {}
        try:
            with open(self.model_signature_file, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _write_signature(self, model_name: str, reindex_pending: bool) -> None:
        try:
            with open(self.model_signature_file, 'w', encoding='utf-8') as f:
                json.dump({"model_name": model_name, "reindex_pending": bool(reindex_pending)}, f)
        except Exception as exc:
            logger.error(f"Nu am putut salva signature: {exc}")

    def get_saved_model_name(self) -> str:
        """Returneaza modelul cu care a fost build-uit cache-ul (sau '' daca lipseste)."""
        return self._read_signature().get("model_name", "")

    def is_reindex_pending(self) -> bool:
        """True daca cache-ul a fost invalidat dar reindexarea DB nu a fost confirmata."""
        return bool(self._read_signature().get("reindex_pending", False))

    def mark_reindex_complete(self) -> None:
        """Reseteaza flag-ul de reindex pending dupa o reindexare completa."""
        self._write_signature(self.model_name, False)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Încarcă metadata din fișier"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.info(f"Metadata încărcată: {len(metadata)} intrări")
                return metadata
            except Exception as e:
                logger.warning(f"Nu s-a putut încărca metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Salvează metadata în fișier"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Eroare la salvarea metadata: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculează hash MD5 pentru un fișier"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Eroare la calcularea hash pentru {file_path}: {e}")
            return ""
    
    def _load_cached_embeddings(self) -> bool:
        """Încarcă embeddings din cache"""
        if not PICKLE_AVAILABLE:
            return False
            
        try:
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                logger.info(f"Embeddings încărcate din cache: {len(self.embeddings_cache)} intrări")
                
            if self.documents_file.exists():
                with open(self.documents_file, 'rb') as f:
                    self.documents_cache = pickle.load(f)
                logger.info(f"Documente încărcate din cache: {len(self.documents_cache)} intrări")
                
            return True
        except Exception as e:
            logger.error(f"Eroare la încărcarea cache-ului: {e}")
            return False
    
    def _save_cached_embeddings(self):
        """Salvează embeddings în cache"""
        if not PICKLE_AVAILABLE:
            return
            
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
                
            with open(self.documents_file, 'wb') as f:
                pickle.dump(self.documents_cache, f)
                
            logger.info("Cache embeddings salvat pe disc")
        except Exception as e:
            logger.error(f"Eroare la salvarea cache-ului: {e}")
    
    def get_cached_embeddings(self, file_paths: List[str]) -> tuple[List[Dict], Optional[Any], bool]:
        """
        Obține embeddings din cache pentru fișierele date
        Returns: (cached_documents, cached_embeddings, all_cached)
        """
        if not self._load_cached_embeddings():
            return [], None, False
        
        cached_documents = []
        cached_embeddings = None
        new_files = []
        
        for file_path in file_paths:
            current_hash = self._calculate_file_hash(file_path)
            file_key = str(Path(file_path).resolve())
            
            # Verifică dacă fișierul este în cache și hash-ul se potrivește
            if (file_key in self.metadata and 
                self.metadata[file_key].get('hash') == current_hash and
                file_key in self.documents_cache and
                file_key in self.embeddings_cache):
                
                file_cached_documents = self.documents_cache[file_key]
                file_cached_embeddings = self.embeddings_cache[file_key]

                if NUMPY_AVAILABLE and hasattr(file_cached_embeddings, "shape"):
                    cached_count = int(file_cached_embeddings.shape[0]) if len(file_cached_embeddings.shape) > 0 else 0
                else:
                    cached_count = len(file_cached_embeddings) if hasattr(file_cached_embeddings, "__len__") else 0

                if cached_count != len(file_cached_documents):
                    logger.warning(
                        f"Cache invalid pentru {Path(file_path).name}: "
                        f"{len(file_cached_documents)} docs vs {cached_count} embeddings. Regenerez fișierul."
                    )
                    new_files.append(file_path)
                    continue

                cached_documents.extend(file_cached_documents)
                if cached_embeddings is None:
                    cached_embeddings = file_cached_embeddings
                else:
                    if NUMPY_AVAILABLE:
                        cached_embeddings = np.vstack([cached_embeddings, file_cached_embeddings])
                    else:
                        cached_embeddings.extend(file_cached_embeddings)
                        
                logger.info(f"Cache hit pentru: {Path(file_path).name}")
            else:
                new_files.append(file_path)
                logger.info(f"Cache miss pentru: {Path(file_path).name}")
        
        all_cached = len(new_files) == 0
        
        logger.info(f"Cache status: {len(cached_documents)} documente din cache, {len(new_files)} fișiere noi")
        
        return cached_documents, cached_embeddings, all_cached
    
    def save_embeddings(self, file_paths: List[str], documents: List[Dict], embeddings: Any):
        """Salvează embeddings în cache pentru fișierele date"""
        if not documents or embeddings is None:
            return
        
        # Grupează documentele după fișier
        files_docs = {}
        files_embeddings = {}
        
        current_idx = 0
        for file_path in file_paths:
            file_key = str(Path(file_path).resolve())
            file_hash = self._calculate_file_hash(file_path)
            
            # Găsește documentele pentru acest fișier
            file_docs = []
            for doc in documents:
                source_path = doc.get('metadata', {}).get('source_path', '')
                if not source_path:
                    continue
                if str(Path(source_path).resolve()) == file_key:
                    file_docs.append(doc)
            doc_count = len(file_docs)
            
            if doc_count > 0:
                # Salvează documentele
                files_docs[file_key] = file_docs
                
                # Salvează embeddings-urile corespunzătoare
                if NUMPY_AVAILABLE and hasattr(embeddings, 'shape'):
                    files_embeddings[file_key] = embeddings[current_idx:current_idx + doc_count]
                else:
                    files_embeddings[file_key] = embeddings[current_idx:current_idx + doc_count]
                
                current_idx += doc_count
                
                # Actualizează metadata
                self.metadata[file_key] = {
                    'hash': file_hash,
                    'timestamp': time.time(),
                    'document_count': doc_count,
                    'file_path': file_path
                }
                
                logger.info(f"Salvat în cache: {Path(file_path).name} ({doc_count} documente)")
        
        # Actualizează cache-urile
        self.documents_cache.update(files_docs)
        self.embeddings_cache.update(files_embeddings)
        
        # Salvează pe disc
        self._save_metadata()
        self._save_cached_embeddings()
        
        logger.info(f"Cache actualizat cu {len(files_docs)} fișiere")
    
    def clear_cache(self):
        """Curată întregul cache"""
        self.metadata = {}
        self.embeddings_cache = {}
        self.documents_cache = {}
        
        # Șterge fișierele de cache
        for file_path in [self.metadata_file, self.embeddings_file, self.documents_file]:
            if file_path.exists():
                file_path.unlink()
        
        logger.info("Cache complet curățat")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Returnează statistici despre cache"""
        total_documents = sum(entry.get('document_count', 0) for entry in self.metadata.values())
        cache_size_mb = 0
        
        for file_path in [self.metadata_file, self.embeddings_file, self.documents_file]:
            if file_path.exists():
                cache_size_mb += file_path.stat().st_size / (1024 * 1024)
        
        return {
            'cached_files': len(self.metadata),
            'total_documents': total_documents,
            'cache_size_mb': round(cache_size_mb, 2),
            'metadata_entries': len(self.metadata)
        }

class MultilingualReranker:
    """Cross-encoder reranker peste retrieved chunks (multilingv).

    Foloseste un model HF de tip cross-encoder care primeste perechi (query, doc)
    si returneaza un scor de relevanta. Modelul se incarca lazy la prima rerank().
    """

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        self.model_name = model_name
        self._model = None
        self._unavailable = not CROSS_ENCODER_AVAILABLE

    @property
    def available(self) -> bool:
        return not self._unavailable

    def _ensure_loaded(self) -> bool:
        if self._unavailable:
            return False
        if self._model is not None:
            return True
        try:
            self._model = CrossEncoder(self.model_name)
            logger.info("Reranker incarcat: %s", self.model_name)
            return True
        except Exception as exc:
            logger.warning("Nu am putut incarca rerankerul %s: %s", self.model_name, exc)
            self._unavailable = True
            return False

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Sorteaza documente dupa relevanta (cross-encoder) si pastreaza primele top_k.

        Pune scorul reranked in `doc["rerank_score"]` si normalizeaza retrieval_score.
        Daca rerankerul nu e disponibil, returneaza documents[:top_k] netratate.
        """
        if not documents:
            return []
        if not self._ensure_loaded():
            return documents[: int(top_k)]

        try:
            pairs = [(query, doc.get("content", "") or "") for doc in documents]
            scores = self._model.predict(pairs)
            scored = list(zip(documents, scores))
            scored.sort(key=lambda item: float(item[1]), reverse=True)

            out: List[Dict[str, Any]] = []
            for doc, score in scored[: int(top_k)]:
                doc["rerank_score"] = float(score)
                # mapam la [0,1] pentru integrare in pipeline-ul de fusion existent
                doc["retrieval_score"] = float(score)
                doc["semantic_score"] = float(score)
                out.append(doc)
            return out
        except Exception as exc:
            logger.warning("Rerank esuat (%s); fallback la ordinea originala.", exc)
            return documents[: int(top_k)]


class OptimizedFlashLLM:
    """LLM optimizat pentru Gemini 2.5 Flash"""
    
    def __init__(self, config: RAGConfig, api_key: str):
        if not GENAI_AVAILABLE:
            raise ImportError("Google GenAI SDK nu este disponibil")
            
        self.config = config
        self.api_key = api_key
        
        # Inițializează clientul Gemini (SDK nou)
        self.client = genai.Client(api_key=api_key)
        
        # Detectează modelul disponibil
        self.model_name = self._detect_best_model()
        
        # Configurare generare
        self.generation_config = genai.types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )
        
        logger.info(f"OptimizedFlashLLM inițializat cu {self.model_name}")
    
    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        if model_name.startswith("models/"):
            return model_name.split("/", 1)[1]
        return model_name

    def _detect_best_model(self) -> str:
        """Detectează cel mai bun model disponibil"""
        try:
            available_models = list(self.client.models.list())
            normalized_available = {
                self._normalize_model_name(model.name)
                for model in available_models
                if getattr(model, "name", "")
            }
            
            # Prioritatea modelelor
            preferred_models = [
                self.config.model_name,
                "gemini-2.5-flash",
                "gemini-2.5-flash-preview-05-20", 
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash",
                "gemini-1.5-flash"
            ]
            
            for model in preferred_models:
                normalized_model = self._normalize_model_name(model)
                if normalized_model in normalized_available:
                    logger.info(f"Model detectat: {normalized_model}")
                    return normalized_model
            
            # Fallback la primul model Flash disponibil
            flash_models = sorted([m for m in normalized_available if "flash" in m.lower()])
            if flash_models:
                logger.warning(f"Folosesc fallback model: {flash_models[0]}")
                return flash_models[0]
            
            raise Exception("Nu s-a găsit niciun model Flash disponibil")
            
        except Exception as e:
            logger.error(f"Eroare la detectarea modelului: {e}")
            return self._normalize_model_name(self.config.fallback_model)
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generează răspuns cu retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.generation_config
                )

                response_text = (response.text or "").strip() if response else ""
                if response_text:
                    return response_text
                else:
                    raise Exception("Răspuns gol de la model")
                    
            except Exception as e:
                logger.warning(f"Tentativa {attempt + 1} eșuată: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Toate tentativele au eșuat pentru prompt: {prompt[:50]}...")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_model_info(self) -> Dict[str, Any]:
        """Returnează informații despre model"""
        return {
            "model_name": self.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "is_flash": "flash" in self.model_name.lower(),
            "optimized_for": "speed_and_efficiency"
        }

class SimpleDocumentProcessor:
    """Procesator simplu de documente"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            self.text_splitter = None
    
    def load_documents(
        self,
        file_paths: List[str],
        metadata_by_path: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Încarcă documente din fișiere"""
        documents = []
        metadata_by_path = metadata_by_path or {}
        
        for file_path in file_paths:
            file_path = Path(file_path)
            resolved_path = str(file_path.resolve())
            extra_metadata = metadata_by_path.get(resolved_path, {})
            
            try:
                text_content = ""
                
                if file_path.suffix.lower() == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                        
                elif file_path.suffix.lower() == '.md':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                        
                elif file_path.suffix.lower() == '.pdf' and LANGCHAIN_AVAILABLE:
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    text_content = "\n".join([doc.page_content for doc in docs]).replace("\x00", "")
                    
                else:
                    logger.warning(f"Tip de fișier nesuportat: {file_path}")
                    continue
                
                if text_content.strip():
                    filename = extra_metadata.get('filename', file_path.name)
                    doc = {
                        'content': text_content,
                        'metadata': {
                            'filename': filename,
                            'file_type': file_path.suffix,
                            'source_path': resolved_path,
                            **extra_metadata
                        }
                    }
                    doc['metadata']['source_path'] = resolved_path
                    documents.append(doc)
                    logger.info(f"Încărcat {filename}")
                
            except Exception as e:
                logger.error(f"Eroare la încărcarea {file_path}: {e}")
        
        return documents
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Procesează documentele cu chunking"""
        if not documents:
            return []
        
        chunks = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc['metadata']
            
            # Chunking simplu dacă LangChain nu e disponibil
            if self.text_splitter:
                # Folosește LangChain text splitter
                lang_doc = Document(page_content=content, metadata=metadata)
                doc_chunks = self.text_splitter.split_documents([lang_doc])
                chunk_counter = 0
                
                for chunk in doc_chunks:
                    if len(chunk.page_content.strip()) > 50:
                        chunk_metadata = dict(chunk.metadata)
                        chunk_metadata["chunk_id"] = chunk_counter
                        chunks.append({
                            'content': chunk.page_content,
                            'metadata': chunk_metadata
                        })
                        chunk_counter += 1
            else:
                # Chunking simplu manual
                chunk_size = self.config.chunk_size
                overlap = self.config.chunk_overlap
                
                start = 0
                chunk_num = 0
                
                while start < len(content):
                    end = start + chunk_size
                    chunk_content = content[start:end]
                    
                    if len(chunk_content.strip()) > 50:
                        chunk_metadata = metadata.copy()
                        chunk_metadata['chunk_id'] = chunk_num
                        
                        chunks.append({
                            'content': chunk_content,
                            'metadata': chunk_metadata
                        })
                        chunk_num += 1
                    
                    start = end - overlap
                    if start >= len(content):
                        break
        
        logger.info(f"Procesat {len(documents)} documente în {len(chunks)} chunks")
        return chunks

class SimpleRetriever:
    """Retriever simplu pentru regăsire de documente cu cache persistent"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.documents = []
        self.embeddings_model = None
        self.document_embeddings = []
        self.chunk_lookup = {}
        
        # Inițializare cache persistent — primeste model_name pentru invalidare automata.
        self.embeddings_cache = PersistentEmbeddingsCache(model_name=config.embedding_model)
        
        # Inițializare model embeddings dacă e disponibil
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embeddings_model = SentenceTransformer(config.embedding_model)
                logger.info(f"Model embeddings încărcat: {config.embedding_model}")
            except Exception as e:
                logger.warning(f"Nu s-a putut încărca modelul embeddings: {e}")

    def _rebuild_chunk_lookup(self):
        """Construiește un index rapid (source_path, chunk_id) -> doc_index."""
        self.chunk_lookup = {}
        for idx, doc in enumerate(self.documents):
            metadata = doc.get("metadata", {})
            source_path = metadata.get("source_path")
            chunk_id = metadata.get("chunk_id")
            if source_path is None or chunk_id is None:
                continue
            try:
                normalized_source = str(Path(source_path).resolve())
                normalized_chunk_id = int(chunk_id)
                self.chunk_lookup[(normalized_source, normalized_chunk_id)] = idx
            except Exception:
                continue
    
    def build_index_with_cache(self, file_paths: List[str], documents: List[Dict[str, Any]]):
        """Construiește indexul cu cache persistent - optimizat pentru viteză"""
        if not documents or not file_paths:
            logger.warning("Nu există documente sau căi de fișiere pentru indexare")
            return
        
        # Verifică cache-ul pentru embeddings existente
        cached_docs, cached_embeddings, all_cached = self.embeddings_cache.get_cached_embeddings(file_paths)
        
        if all_cached:
            # Toate documentele sunt în cache - încărcare instantanee!
            logger.info("CACHE HIT COMPLET - Încărcare instantanee din cache!")
            
            if not hasattr(self, 'documents') or self.documents is None:
                self.documents = []
            
            # Verifică pentru duplicate
            existing_contents = {doc['content'] for doc in self.documents} if self.documents else set()
            new_cached_docs = [doc for doc in cached_docs if doc['content'] not in existing_contents]
            
            if new_cached_docs:
                self.documents.extend(new_cached_docs)
                
                # Adaugă embeddings din cache
                if cached_embeddings is not None:
                    if hasattr(self, 'document_embeddings') and len(self.document_embeddings) > 0:
                        if NUMPY_AVAILABLE:
                            self.document_embeddings = np.vstack([self.document_embeddings, cached_embeddings])
                        else:
                            self.document_embeddings = np.concatenate([self.document_embeddings, cached_embeddings])
                    else:
                        self.document_embeddings = cached_embeddings
                
                logger.info(f"Încărcat instant din cache: {len(new_cached_docs)} documente")
            else:
                logger.info("Toate documentele din cache sunt deja încărcate")
            self._rebuild_chunk_lookup()
            return
        
        # Cache parțial sau lipsă - procesare necesară
        logger.info(f"Cache parțial: {len(cached_docs)} din cache, {len(documents) - len(cached_docs)} de procesat")
        
        # Încărcare incrementală: adaugă la documentele existente
        if not hasattr(self, 'documents') or self.documents is None:
            self.documents = []
        
        # Adaugă documentele din cache mai întâi
        if cached_docs:
            existing_contents = {doc['content'] for doc in self.documents} if self.documents else set()
            new_cached_docs = [doc for doc in cached_docs if doc['content'] not in existing_contents]
            
            if new_cached_docs:
                self.documents.extend(new_cached_docs)
                
                if cached_embeddings is not None:
                    if hasattr(self, 'document_embeddings') and len(self.document_embeddings) > 0:
                        if NUMPY_AVAILABLE:
                            self.document_embeddings = np.vstack([self.document_embeddings, cached_embeddings])
                        else:
                            self.document_embeddings = np.concatenate([self.document_embeddings, cached_embeddings])
                    else:
                        self.document_embeddings = cached_embeddings
                
                logger.info(f"Adăugat din cache: {len(new_cached_docs)} documente")
        
        # Procesează doar documentele noi (care nu sunt în cache)
        existing_contents = {doc['content'] for doc in self.documents} if self.documents else set()
        new_docs = [doc for doc in documents if doc['content'] not in existing_contents]
        
        if new_docs:
            self.documents.extend(new_docs)
            logger.info(f"Procesez {len(new_docs)} documente noi")
            
            # Generează embeddings doar pentru documentele noi
            if self.embeddings_model:
                try:
                    new_contents = [doc['content'] for doc in new_docs]
                    new_embeddings = self.embeddings_model.encode(new_contents)
                    
                    # Concatenează cu embeddings existente
                    if hasattr(self, 'document_embeddings') and len(self.document_embeddings) > 0:
                        if NUMPY_AVAILABLE:
                            self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])
                        else:
                            self.document_embeddings = np.concatenate([self.document_embeddings, new_embeddings])
                    else:
                        self.document_embeddings = new_embeddings
                    
                    logger.info(f"Generat embeddings pentru {len(new_docs)} documente noi")
                    
                    # Salvează în cache pentru viitorul utilizări
                    self.embeddings_cache.save_embeddings(file_paths, new_docs, new_embeddings)
                    
                except Exception as e:
                    logger.error(f"Eroare la generarea embeddings: {e}")
                    self.document_embeddings = []
        else:
            logger.info("Toate documentele sunt deja în index")
        
        self._rebuild_chunk_lookup()
        logger.info(f"Index finalizat cu {len(self.documents)} documente totale")
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Construiește indexul pentru regăsire - versiunea originală pentru compatibilitate"""
        if not documents:
            logger.warning("Nu există documente pentru indexare")
            return
        
        # Încărcare incrementală: adaugă la documentele existente în loc să le suprascrie
        if not hasattr(self, 'documents') or self.documents is None:
            self.documents = []
        
        # Verifică pentru duplicate (opțional)
        existing_contents = {doc['content'] for doc in self.documents} if self.documents else set()
        new_docs = [doc for doc in documents if doc['content'] not in existing_contents]
        
        if new_docs:
            self.documents.extend(new_docs)
            logger.info(f"Adăugat {len(new_docs)} documente noi la indexul existent de {len(self.documents) - len(new_docs)} documente")
        else:
            logger.info("Toate documentele sunt deja în index")
        
        if self.embeddings_model:
            try:
                # Generează embeddings doar pentru documentele noi sau regenerează toate
                if new_docs and hasattr(self, 'document_embeddings') and len(self.document_embeddings) > 0:
                    # Adaugă embeddings pentru documentele noi
                    new_contents = [doc['content'] for doc in new_docs]
                    new_embeddings = self.embeddings_model.encode(new_contents)
                    
                    # Concatenează cu embeddings existente
                    import numpy as np
                    if NUMPY_AVAILABLE and len(self.document_embeddings) > 0:
                        self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings])
                    else:
                        self.document_embeddings = new_embeddings
                    
                    logger.info(f"Adăugat embeddings pentru {len(new_docs)} documente noi")
                else:
                    # Generează embeddings pentru toate documentele (prima încărcare sau regenerare)
                    contents = [doc['content'] for doc in self.documents]
                    self.document_embeddings = self.embeddings_model.encode(contents)
                    logger.info(f"Generat embeddings pentru toate {len(self.documents)} documentele")
            except Exception as e:
                logger.error(f"Eroare la generarea embeddings: {e}")
                self.document_embeddings = []
        
        self._rebuild_chunk_lookup()
        logger.info(f"Index actualizat cu {len(self.documents)} documente totale")
    
    def retrieve(
        self,
        query: str,
        k: int = None,
        filter_fn: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Regăsește documente relevante cu scoring hibrid și extindere pe vecini."""
        if not self.documents:
            logger.warning("Nu există documente indexate")
            return []
        
        k = k or self.config.max_context_docs
        candidate_indices = list(range(len(self.documents)))
        if filter_fn:
            candidate_indices = [
                idx for idx, doc in enumerate(self.documents) if filter_fn(doc)
            ]
        
        if not candidate_indices:
            logger.info("Nu există documente candidate pentru filtrul curent")
            return []

        semantic_scores = self._compute_semantic_scores(query, candidate_indices)
        keyword_scores = self._compute_keyword_scores(query, candidate_indices)

        semantic_norm = self._normalize_scores(semantic_scores, candidate_indices)
        keyword_norm = self._normalize_scores(keyword_scores, candidate_indices)

        has_semantic = bool(semantic_scores)
        alpha = self.config.hybrid_alpha if has_semantic else 0.0
        alpha = max(0.0, min(alpha, 1.0))

        hybrid_scores = {}
        for idx in candidate_indices:
            hybrid_scores[idx] = (
                alpha * semantic_norm.get(idx, 0.0)
                + (1.0 - alpha) * keyword_norm.get(idx, 0.0)
            )

        candidate_pool = max(k, self.config.retrieval_candidates)
        candidate_pool = min(candidate_pool, len(candidate_indices))
        ranked_candidates = sorted(
            candidate_indices,
            key=lambda idx: hybrid_scores.get(idx, 0.0),
            reverse=True
        )[:candidate_pool]

        seed_k = max(k, self.config.rerank_top_k)
        seed_k = min(seed_k, len(ranked_candidates))
        seed_indices = ranked_candidates[:seed_k]

        expanded_scores = self._expand_with_neighbors(seed_indices, hybrid_scores, candidate_indices)
        final_indices = sorted(
            expanded_scores.keys(),
            key=lambda idx: expanded_scores.get(idx, 0.0),
            reverse=True
        )[:k]

        results = []
        for idx in final_indices:
            doc = self.documents[idx]
            results.append({
                "content": doc["content"],
                "metadata": doc.get("metadata", {}).copy(),
                "retrieval_score": float(expanded_scores.get(idx, 0.0)),
                "semantic_score": float(semantic_norm.get(idx, 0.0)),
                "keyword_score": float(keyword_norm.get(idx, 0.0))
            })

        return results

    @staticmethod
    def _normalize_scores(score_map: Dict[int, float], candidate_indices: List[int]) -> Dict[int, float]:
        if not candidate_indices:
            return {}

        values = [float(score_map.get(idx, 0.0)) for idx in candidate_indices]
        if not values:
            return {idx: 0.0 for idx in candidate_indices}

        minimum = min(values)
        maximum = max(values)
        if abs(maximum - minimum) < 1e-9:
            return {idx: 1.0 if score_map.get(idx, 0.0) > 0 else 0.0 for idx in candidate_indices}

        return {
            idx: (float(score_map.get(idx, 0.0)) - minimum) / (maximum - minimum)
            for idx in candidate_indices
        }

    def _compute_semantic_scores(self, query: str, candidate_indices: List[int]) -> Dict[int, float]:
        if not (
            self.embeddings_model
            and len(self.document_embeddings) > 0
            and len(self.document_embeddings) == len(self.documents)
        ):
            return {}

        try:
            query_embedding = self.embeddings_model.encode([query])

            if NUMPY_AVAILABLE:
                if hasattr(self.document_embeddings, "shape"):
                    candidate_embeddings = self.document_embeddings[candidate_indices]
                else:
                    candidate_embeddings = np.array([self.document_embeddings[idx] for idx in candidate_indices])
                similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()
            else:
                similarities = []
                for index in candidate_indices:
                    doc_emb = self.document_embeddings[index]
                    sim = sum(a * b for a, b in zip(doc_emb, query_embedding[0]))
                    similarities.append(sim)

            return {
                candidate_indices[position]: float(score)
                for position, score in enumerate(similarities)
            }
        except Exception as e:
            logger.error(f"Eroare la scor semantic: {e}")
            return {}

    def _compute_keyword_scores(self, query: str, candidate_indices: List[int]) -> Dict[int, float]:
        query_words = set(query.lower().split())
        if not query_words:
            return {idx: 0.0 for idx in candidate_indices}

        scores = {}
        for index in candidate_indices:
            doc = self.documents[index]
            content_words = set(doc['content'].lower().split())
            word_overlap = len(query_words.intersection(content_words))
            density_score = word_overlap / max(len(query_words), 1)
            length_penalty = min(len(doc['content']) / 2000.0, 1.0)
            scores[index] = float(density_score + 0.15 * length_penalty)
        return scores

    def _expand_with_neighbors(
        self,
        seed_indices: List[int],
        base_scores: Dict[int, float],
        candidate_indices: List[int]
    ) -> Dict[int, float]:
        expanded_scores = {idx: float(base_scores.get(idx, 0.0)) for idx in seed_indices}
        candidate_set = set(candidate_indices)
        window = max(0, int(self.config.neighbor_window))
        if window == 0:
            return expanded_scores

        for seed_idx in seed_indices:
            seed_doc = self.documents[seed_idx]
            seed_meta = seed_doc.get("metadata", {})
            source_path = seed_meta.get("source_path")
            chunk_id = seed_meta.get("chunk_id")
            if source_path is None or chunk_id is None:
                continue

            try:
                normalized_source = str(Path(source_path).resolve())
                normalized_chunk_id = int(chunk_id)
            except Exception:
                continue

            seed_score = float(base_scores.get(seed_idx, 0.0))
            for offset in range(1, window + 1):
                for neighbor_chunk_id in (normalized_chunk_id - offset, normalized_chunk_id + offset):
                    neighbor_idx = self.chunk_lookup.get((normalized_source, neighbor_chunk_id))
                    if neighbor_idx is None or neighbor_idx not in candidate_set:
                        continue
                    neighbor_base = float(base_scores.get(neighbor_idx, 0.0))
                    boosted_score = max(neighbor_base, seed_score * (1.0 - 0.08 * offset))
                    previous = expanded_scores.get(neighbor_idx, -1.0)
                    if boosted_score > previous:
                        expanded_scores[neighbor_idx] = boosted_score

        return expanded_scores

class CerebrumAISystem:
    """Sistemul principal CerebrumAI cu toate optimizările"""
    
    def __init__(self, config: RAGConfig, api_key: str):
        self.config = config
        self.api_key = api_key
        
        # Inițializare componente
        self.llm = OptimizedFlashLLM(config, api_key)
        self.rate_limiter = GeminiRateLimiter(
            requests_per_minute=config.rate_limit_rpm,
            requests_per_day=config.rate_limit_rpd
        )
        
        # Încearcă să folosească cache avansat sau fallback
        if DISKCACHE_AVAILABLE and config.cache_enabled:
            try:
                from diskcache import Cache
                cache_dir = Path("./data/cache")
                cache_dir.mkdir(parents=True, exist_ok=True)
                self.cache_manager = Cache(str(cache_dir / "responses_cache"))
                logger.info("Cache avansat (diskcache) activat")
            except Exception as e:
                logger.warning(f"Nu s-a putut inițializa cache avansat: {e}")
                self.cache_manager = SimpleCache() if config.cache_enabled else None
        else:
            self.cache_manager = SimpleCache() if config.cache_enabled else None
        
        self.document_processor = SimpleDocumentProcessor(config)
        self.retriever = SimpleRetriever(config)

        # Reranker multilingv (P3) — lazy load la prima utilizare.
        self.reranker: Optional[MultilingualReranker] = None
        if config.enable_reranker and CROSS_ENCODER_AVAILABLE:
            try:
                self.reranker = MultilingualReranker(model_name=config.reranker_model)
                logger.info("Reranker pregatit (lazy load): %s", config.reranker_model)
            except Exception as exc:
                logger.warning("Init reranker esuat: %s", exc)
                self.reranker = None

        # Second Brain storage (Postgres + pgvector), optional and fallback-safe.
        self.storage_client = None
        self.repository = None
        if STORAGE_AVAILABLE:
            try:
                self.storage_client = PostgresClient(
                    dsn=config.postgres_dsn,
                    embedding_dim=config.pgvector_embedding_dim,
                    enabled=True,
                )
                self.repository = SecondBrainRepository(self.storage_client)
                if self.repository.enabled:
                    ready = self.repository.ensure_ready()
                    logger.info("Storage backend Postgres activ: %s", ready)
            except Exception as exc:
                logger.warning("Storage backend init failed, fallback local: %s", exc)
                self.storage_client = None
                self.repository = None

        # Graph layer (Neo4j), optional and fallback-safe.
        self.neo4j_client = None
        self.graph_ingestion = None
        self.graph_query = None
        if GRAPH_AVAILABLE and config.enable_graph_rag:
            try:
                self.neo4j_client = Neo4jClient(
                    uri=config.neo4j_uri,
                    user=config.neo4j_user,
                    password=config.neo4j_password,
                    enabled=True,
                )
                self.graph_ingestion = GraphIngestionService(
                    self.neo4j_client,
                    min_confidence=config.graph_min_confidence,
                    extraction_mode=config.graph_extraction_mode,
                    structured_extractor=self._extract_graph_structure_with_llm,
                )
                self.graph_query = GraphQueryService(
                    self.neo4j_client,
                    max_hops=config.graph_max_hops,
                )
                if self.neo4j_client.enabled:
                    self.neo4j_client.ensure_constraints()
                    logger.info("Graph backend Neo4j activ.")
            except Exception as exc:
                logger.warning("Graph backend init failed, fallback vector-only: %s", exc)
                self.neo4j_client = None
                self.graph_ingestion = None
                self.graph_query = None

        # Query intelligence components.
        self.intent_router = None
        self.hybrid_fusion = None
        self.context_builder = None
        if QUERY_INTEL_AVAILABLE:
            self.intent_router = RuleBasedIntentRouter(
                vector_confidence_threshold=config.vector_confidence_threshold
            )
            self.hybrid_fusion = HybridEvidenceFusion()
            self.context_builder = ContextBuilder(
                total_budget_tokens=config.context_budget_tokens,
                vector_budget_ratio=config.vector_budget_ratio,
                graph_budget_ratio=config.graph_budget_ratio,
            )

        # Local fallback memory when DB is not configured.
        self.local_decisions: List[Dict[str, Any]] = []
        self.local_preferences: List[Dict[str, Any]] = []
        self.local_episodes: List[Dict[str, Any]] = []
        self.local_tasks: List[Dict[str, Any]] = []
        self.last_memory_consolidation_at: Optional[str] = None
        
        # Statistici
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "avg_response_time": 0,
            "documents_indexed": 0
        }
        
        logger.info("CerebrumAISystem inițializat cu succes")
    
    def load_documents(
        self,
        file_paths: List[str],
        metadata_by_path: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> bool:
        """Încarcă și indexează documente cu cache persistent pentru viteză optimă"""
        try:
            start_time = time.time()
            resolved_file_paths = [str(Path(path).resolve()) for path in file_paths]
            
            # Încarcă documente
            documents = self.document_processor.load_documents(
                resolved_file_paths,
                metadata_by_path=metadata_by_path
            )
            if not documents:
                logger.warning("Nu s-au încărcat documente")
                return False
            
            # Procesează documente
            processed_docs = self.document_processor.process_documents(documents)
            if not processed_docs:
                logger.warning("Nu s-au procesat documente")
                return False
            
            # Construiește index cu cache persistent - aceasta este optimizarea cheie!
            self.retriever.build_index_with_cache(resolved_file_paths, processed_docs)

            # Persistență Postgres + pgvector (opțional, fără a bloca flow-ul curent).
            if self.repository and self.repository.enabled:
                storage_embeddings = None
                try:
                    if self.retriever.embeddings_model:
                        storage_embeddings = self._encode_embeddings_batched(
                            [doc["content"] for doc in processed_docs]
                        )
                except Exception as exc:
                    logger.warning("Embeddings storage generation failed: %s", exc)
                    storage_embeddings = None

                self.repository.ingest_processed_documents(
                    resolved_file_paths,
                    processed_docs,
                    storage_embeddings,
                )

            # Ingestie incrementală în graf (offline-style).
            if self.graph_ingestion and self.graph_ingestion.enabled:
                self.graph_ingestion.ingest_chunks(processed_docs)
            
            self.stats["documents_indexed"] = len(processed_docs)
            
            end_time = time.time()
            load_time = end_time - start_time
            
            logger.info(f"Încărcare completă în {load_time:.2f}s: {len(processed_docs)} chunks din {len(file_paths)} fișiere")
            
            return True
            
        except Exception as e:
            logger.error(f"Eroare la încărcarea documentelor: {e}")
            return False

    def _encode_embeddings_batched(self, texts: List[str]) -> Any:
        """Encode embeddings in batches to avoid memory spikes on large ingestions."""
        if not texts or not self.retriever.embeddings_model:
            return None

        batch_size = max(8, int(self.config.ingestion_embedding_batch_size or 64))
        encoded_batches: List[Any] = []
        for start_idx in range(0, len(texts), batch_size):
            batch = texts[start_idx:start_idx + batch_size]
            encoded_batch = self.retriever.embeddings_model.encode(batch)
            encoded_batches.append(encoded_batch)

        if not encoded_batches:
            return None

        if NUMPY_AVAILABLE:
            return np.vstack(encoded_batches)

        merged: List[Any] = []
        for batch in encoded_batches:
            merged.extend(batch)
        return merged

    @staticmethod
    def _calculate_file_hash(file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def filter_paths_for_ingestion(
        self,
        file_paths: List[str],
        metadata_by_path: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Return ingestion candidates using Postgres as source-of-truth.
        When storage is disabled/unavailable, all files are considered candidates.
        """
        resolved_paths: List[str] = []
        seen = set()
        for path in file_paths:
            normalized = str(Path(path).resolve())
            if normalized in seen:
                continue
            seen.add(normalized)
            resolved_paths.append(normalized)

        if not resolved_paths:
            return {"to_ingest": [], "already_indexed": []}

        if not (self.repository and self.repository.enabled):
            return {"to_ingest": resolved_paths, "already_indexed": []}

        try:
            metadata_by_path = metadata_by_path or {}
            hash_by_path: Dict[str, str] = {}
            paths_without_hash: List[str] = []

            for path in resolved_paths:
                metadata = metadata_by_path.get(path, {})
                file_hash = (metadata.get("file_hash") or "").strip() if isinstance(metadata, dict) else ""
                if file_hash:
                    hash_by_path[path] = file_hash
                else:
                    paths_without_hash.append(path)

            if paths_without_hash:
                max_workers = max(1, int(self.config.ingestion_max_workers or 4))
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_map = {
                        pool.submit(self._calculate_file_hash, path): path
                        for path in paths_without_hash
                    }
                    for future in as_completed(future_map):
                        path = future_map[future]
                        try:
                            hash_by_path[path] = future.result()
                        except Exception as exc:
                            logger.warning("Hash generation failed for %s: %s", path, exc)
                            hash_by_path[path] = ""

            missing_hashes = set(
                self.repository.list_missing_hashes([value for value in hash_by_path.values() if value])
            )
            to_ingest: List[str] = []
            already_indexed: List[str] = []

            for path in resolved_paths:
                file_hash = hash_by_path.get(path, "")
                if file_hash and file_hash in missing_hashes:
                    to_ingest.append(path)
                else:
                    already_indexed.append(path)

            return {"to_ingest": to_ingest, "already_indexed": already_indexed}
        except Exception as exc:
            logger.warning("Nu s-a putut filtra ingestia pe baza storage: %s", exc)
            return {"to_ingest": resolved_paths, "already_indexed": []}

    def get_storage_index_status(self) -> Dict[str, Any]:
        """Return storage index status for UI bootstrap decisions."""
        base_status = {
            "storage_mode": "local_fallback",
            "postgres_enabled": False,
            "indexed_documents": 0,
            "indexed_chunks": 0,
            "last_sync_at": None,
            "db_primary_strict": bool(self.config.db_primary_strict),
        }
        if not (self.repository and self.repository.enabled):
            return base_status

        index_status = self.repository.get_index_status()
        return {
            "storage_mode": "db_primary",
            "postgres_enabled": True,
            "indexed_documents": int(index_status.get("indexed_documents", 0) or 0),
            "indexed_chunks": int(index_status.get("indexed_chunks", 0) or 0),
            "last_sync_at": index_status.get("last_sync_at"),
            "db_primary_strict": bool(self.config.db_primary_strict),
        }
    
    def load_documents_legacy(self, file_paths: List[str]) -> bool:
        """Încarcă și indexează documente - versiunea originală fără cache"""
        try:
            # Încarcă documente
            resolved_file_paths = [str(Path(path).resolve()) for path in file_paths]
            documents = self.document_processor.load_documents(resolved_file_paths)
            if not documents:
                logger.warning("Nu s-au încărcat documente")
                return False
            
            # Procesează documente
            processed_docs = self.document_processor.process_documents(documents)
            if not processed_docs:
                logger.warning("Nu s-au procesat documente")
                return False
            
            # Construiește index (metoda originală)
            self.retriever.build_index(processed_docs)
            
            self.stats["documents_indexed"] = len(processed_docs)
            logger.info(f"Încărcat și indexat {len(processed_docs)} chunks din {len(file_paths)} fișiere")
            
            return True
            
        except Exception as e:
            logger.error(f"Eroare la încărcarea documentelor: {e}")
            return False
    
    def _create_rag_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Creează prompt optimizat pentru RAG"""
        if not context_docs:
            return f"""Ești CerebrumAI (Asistentul Personalizat de Cercetare și Învățare), un AI expert în educație și cercetare.

Întrebare: {query}

Dacă nu ai context suficient, răspunde exact: INSUFFICIENT_CONTEXT.
În caz contrar, răspunde concis și informativ, oferind informații relevante și practice.

Răspuns:"""
        
        # Creează context compact
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            filename = doc['metadata'].get('filename', f'Doc{i}')
            # Limitează fiecare document pentru eficiență
            content = doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
            context_parts.append(f"[{i}. {filename}]: {content}")
        
        context = "\n".join(context_parts)
        
        return f"""Ești CerebrumAI (Asistentul Personalizat de Cercetare și Învățare). Analizează contextul și răspunde la întrebare.

CONTEXT:
{context}

ÎNTREBARE: {query}

INSTRUCȚIUNI:
- Răspunde pe baza contextului furnizat
- Citează sursele relevante [număr]
- Dacă contextul este insuficient, răspunde exact: INSUFFICIENT_CONTEXT

RĂSPUNS:"""
    
    @staticmethod
    def _normalize_collection_name(name: Optional[str]) -> str:
        return (name or "").strip().lower()

    def _is_general_collection(self, collection_name: Optional[str], general_collection: str) -> bool:
        normalized = self._normalize_collection_name(collection_name)
        general_aliases = {"general", "default", self._normalize_collection_name(general_collection)}
        return normalized in general_aliases

    def _build_scope_filter(
        self,
        retrieval_mode: str,
        active_collection: Optional[str],
        general_collection: str
    ):
        normalized_mode = (retrieval_mode or "topic_general").strip().lower()
        normalized_active = self._normalize_collection_name(active_collection)

        if normalized_mode == "all":
            return None

        if normalized_mode == "topic":
            if not normalized_active:
                return lambda _: False
            return lambda doc: self._normalize_collection_name(
                doc.get("metadata", {}).get("collection", "general")
            ) == normalized_active

        if normalized_mode == "topic_general":
            if not normalized_active:
                return lambda doc: self._is_general_collection(
                    doc.get("metadata", {}).get("collection", "general"),
                    general_collection
                )
            return lambda doc: (
                self._normalize_collection_name(doc.get("metadata", {}).get("collection", "general"))
                == normalized_active
                or self._is_general_collection(
                    doc.get("metadata", {}).get("collection", "general"),
                    general_collection
                )
            )

        return None

    def _resolve_collection_filters(
        self,
        retrieval_mode: str,
        active_collection: Optional[str],
        general_collection: str
    ) -> Optional[List[str]]:
        normalized_mode = (retrieval_mode or "topic_general").strip().lower()
        normalized_active = (active_collection or "").strip()
        normalized_general = (general_collection or "general").strip()

        if normalized_mode == "all":
            return None
        if normalized_mode == "topic":
            return [normalized_active] if normalized_active else []
        if normalized_mode in {"topic_general", "auto", ""}:
            if normalized_active:
                return [normalized_active, normalized_general]
            return [normalized_general]
        return None

    @staticmethod
    def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text:
            return None
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    def _extract_graph_structure_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Structured concept/relation extraction for GraphRAG ingestion.
        Returns payload:
        {
          "concepts": [{"name": "...", "confidence": 0.0-1.0}],
          "relations": [{"source": "...", "target": "...", "type": "...", "confidence": 0.0-1.0}]
        }
        """
        if not text or not self.config.enable_graph_rag:
            return {"concepts": [], "relations": []}

        snippet = text[:2500]
        prompt = (
            "Extract a compact knowledge graph from the text.\n"
            "Return STRICT JSON only (no markdown, no explanations).\n"
            "Schema:\n"
            "{\n"
            '  "concepts": [{"name": "string", "confidence": 0.0}],\n'
            '  "relations": [{"source": "string", "target": "string", "type": "RELATED_TO|DEPENDS_ON|CONTRADICTS|DERIVED_FROM", "confidence": 0.0}]\n'
            "}\n"
            "Rules:\n"
            "- Max 12 concepts and max 16 relations.\n"
            "- Use concise canonical names.\n"
            "- confidence must be in [0,1].\n"
            "- Only include relations grounded in text.\n"
            f"TEXT:\n{snippet}"
        )
        try:
            self.rate_limiter.wait_if_needed()
            self.rate_limiter.add_request()
            raw = self.llm.generate(prompt)
            self.stats["llm_calls"] += 1
            payload = self._extract_json_object(raw) or {}
            concepts = payload.get("concepts", [])
            relations = payload.get("relations", [])
            if not isinstance(concepts, list):
                concepts = []
            if not isinstance(relations, list):
                relations = []
            return {"concepts": concepts, "relations": relations}
        except Exception as exc:
            logger.warning("LLM graph extraction failed: %s", exc)
            return {"concepts": [], "relations": []}

    def _run_vector_retrieval(
        self,
        question: str,
        scope_filter: Optional[Any],
        collection_filters: Optional[List[str]],
        k: int
    ) -> List[Dict[str, Any]]:
        final_k = max(int(k), int(self.config.vector_top_k))

        # Daca rerankerul e activ, oversample din DB ca sa avem material pentru rerank.
        rerank_active = bool(
            self.reranker
            and self.reranker.available
            and self.config.enable_reranker
        )
        retrieval_k = max(final_k, int(self.config.reranker_input_k)) if rerank_active else final_k

        # Optional HyDE: medie embeddings (query + pseudo-doc generat)
        query_embedding = self._compute_query_embedding(question)

        # Prefer Postgres vector retrieval if configured.
        if (
            self.repository
            and self.repository.enabled
            and query_embedding is not None
        ):
            try:
                db_docs = self.repository.vector_search(
                    query_embedding=query_embedding,
                    collection_filters=collection_filters,
                    top_k=retrieval_k,
                )
                if db_docs:
                    return self._maybe_rerank(question, db_docs, final_k)

                # In strict DB-primary mode, empty DB results are final; avoid stale local fallback.
                if self.config.db_primary_strict:
                    return []
            except Exception as exc:
                logger.warning("DB vector retrieval failed; fallback to in-memory retriever: %s", exc)

        # Fallback to in-memory retriever.
        local_docs = self.retriever.retrieve(
            question,
            k=retrieval_k,
            filter_fn=scope_filter,
        )
        return self._maybe_rerank(question, local_docs, final_k)

    def _maybe_rerank(
        self,
        question: str,
        documents: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Aplica rerankerul daca e activ; altfel returneaza top_k din lista originala."""
        if not documents:
            return []
        if not (self.reranker and self.reranker.available and self.config.enable_reranker):
            return documents[: int(top_k)]
        return self.reranker.rerank(question, documents, top_k=top_k)

    def _compute_query_embedding(self, question: str) -> Optional[List[float]]:
        """Calculeaza embedding-ul query-ului. Daca HyDE e activ, mediaza cu pseudo-doc."""
        if self.retriever.embeddings_model is None:
            return None
        try:
            base_emb = self.retriever.embeddings_model.encode([question])[0]
            if hasattr(base_emb, "tolist"):
                base_emb = base_emb.tolist()

            if self.config.enable_hyde:
                pseudo_doc = self._generate_hyde_pseudo_doc(question)
                if pseudo_doc:
                    pseudo_emb = self.retriever.embeddings_model.encode([pseudo_doc])[0]
                    if hasattr(pseudo_emb, "tolist"):
                        pseudo_emb = pseudo_emb.tolist()
                    if NUMPY_AVAILABLE:
                        avg = ((np.asarray(base_emb) + np.asarray(pseudo_emb)) / 2.0).tolist()
                        return avg
                    # fallback fara numpy
                    return [(b + p) / 2.0 for b, p in zip(base_emb, pseudo_emb)]
            return base_emb
        except Exception as exc:
            logger.warning("Query embedding esuat: %s", exc)
            return None

    def _generate_hyde_pseudo_doc(self, question: str) -> str:
        """Genereaza un raspuns ipotetic scurt cu Gemini, util pentru retrieve (HyDE)."""
        if not question or len(question.strip()) < 4:
            return ""
        prompt = (
            "Imagine that you are answering the following question with a 1-2 sentence "
            "factual passage in the same language as the question. Output only the passage, "
            "without explanation or preface.\n\n"
            f"Question: {question.strip()}\n\nPassage:"
        )
        try:
            self.rate_limiter.wait_if_needed()
            self.rate_limiter.add_request()
            raw = self.llm.generate(
                prompt,
                max_output_tokens=int(self.config.hyde_max_tokens),
            )
            self.stats["llm_calls"] += 1
            return (raw or "").strip()
        except TypeError:
            # llm.generate fara max_output_tokens
            try:
                raw = self.llm.generate(prompt)
                return (raw or "").strip()
            except Exception as exc:
                logger.warning("HyDE generation failed: %s", exc)
                return ""
        except Exception as exc:
            logger.warning("HyDE generation failed: %s", exc)
            return ""

    def _run_graph_retrieval(
        self,
        question: str,
        active_collection: Optional[str],
        max_paths: int
    ) -> List[Dict[str, Any]]:
        if not self.graph_query or not self.graph_query.enabled:
            return []
        try:
            return self.graph_query.retrieve_paths(
                question=question,
                active_collection=active_collection,
                max_paths=max_paths,
            )
        except Exception as exc:
            logger.warning("Graph retrieval failed: %s", exc)
            return []

    def _build_route_context(
        self,
        question: str,
        route: str,
        vector_docs: List[Dict[str, Any]],
        graph_paths: List[Dict[str, Any]],
        memory_hits: Optional[List[Dict[str, Any]]] = None,
        note_hits: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        memory_hits = memory_hits or []
        note_hits = note_hits or []

        memory_lines: List[str] = []
        memory_provenance: List[Dict[str, Any]] = []
        for idx, item in enumerate(memory_hits[:3], start=1):
            title = item.get("title", f"Memorie {idx}")
            rationale = item.get("rationale", "")
            topic = item.get("topic_collection", "")
            memory_type = item.get("memory_type", "semantic")
            citation = f"M{idx}"
            memory_preview = rationale[:300] + "..." if isinstance(rationale, str) and len(rationale) > 300 else rationale
            memory_lines.append(f"[{citation}] [{memory_type}] {title} ({topic}) -> {memory_preview}")
            memory_provenance.append(
                {
                    "citation": citation,
                    "kind": "memory",
                    "score": float(item.get("confidence", 0.0) or 0.0),
                    "source": item,
                }
            )

        note_lines: List[str] = []
        note_provenance: List[Dict[str, Any]] = []
        for idx, note in enumerate(note_hits[:3], start=1):
            note_title = note.get("title") or f"Nota {idx}"
            note_content = note.get("content", "")
            note_topic = note.get("topic_collection", "") or "global"
            citation = f"N{idx}"
            preview = note_content[:400] + "..." if len(note_content) > 400 else note_content
            note_lines.append(f"[{citation}] {note_title} ({note_topic}) -> {preview}")
            note_provenance.append(
                {
                    "citation": citation,
                    "kind": "note",
                    "score": float(note.get("similarity", 0.0) or 0.0),
                    "source": note,
                }
            )

        if (
            self.context_builder is None
            or self.hybrid_fusion is None
            or not QUERY_INTEL_AVAILABLE
        ):
            prompt = self._create_rag_prompt(question, vector_docs[: self.config.max_context_docs])
            if memory_lines:
                prompt += (
                    "\n\nMEMORIE PERSONALĂ:\n"
                    + "\n".join(memory_lines)
                    + "\nFolosește citări [M#] când utilizezi memorie personală."
                )
            if note_lines:
                prompt += (
                    "\n\nNOTELE TALE PERSONALE:\n"
                    + "\n".join(note_lines)
                    + "\nFolosește citări [N#] când utilizezi notele tale."
                )
            return {
                "prompt": prompt,
                "provenance": memory_provenance + note_provenance,
                "vector_sources": [doc.get("metadata", {}) for doc in vector_docs[: self.config.max_context_docs]],
                "graph_sources": [],
                "memory_sources": memory_hits[:3],
                "note_sources": note_hits[:3],
                "selected_vector_docs": vector_docs[: self.config.max_context_docs],
            }

        if route == "hybrid":
            evidence_items, provenance = self.hybrid_fusion.fuse(
                vector_docs=vector_docs,
                graph_paths=graph_paths,
                rerank_top_k=self.config.hybrid_rerank_top_k,
            )
        else:
            evidence_items, provenance = self.hybrid_fusion.fuse(
                vector_docs=vector_docs,
                graph_paths=[],
                rerank_top_k=self.config.max_context_docs,
            )

        built = self.context_builder.build(question, evidence_items)
        built["provenance"] = provenance if provenance else built.get("provenance", [])
        if memory_lines:
            built["prompt"] = (
                built.get("prompt", "")
                + "\n\nMEMORIE PERSONALĂ:\n"
                + "\n".join(memory_lines)
                + "\nFolosește citări [M#] când utilizezi memorie personală."
            )
            built["provenance"].extend(memory_provenance)
        if note_lines:
            built["prompt"] = (
                built.get("prompt", "")
                + "\n\nNOTELE TALE PERSONALE:\n"
                + "\n".join(note_lines)
                + "\nFolosește citări [N#] când utilizezi notele tale."
            )
            built["provenance"].extend(note_provenance)
        built["memory_sources"] = memory_hits[:3]
        built["note_sources"] = note_hits[:3]

        selected_vector_docs = []
        for item in evidence_items:
            if item.kind != "document":
                continue
            selected_vector_docs.append(
                {
                    "content": item.content,
                    "metadata": item.source_ref,
                    "retrieval_score": item.score,
                }
            )
        built["selected_vector_docs"] = selected_vector_docs
        return built

    def _retrieve_memory_hits(
        self,
        question: str,
        active_collection: Optional[str],
        limit: int = 5,
        time_window_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        memory_hits: List[Dict[str, Any]] = []
        dedup_ids = set()

        def _push(items: List[Dict[str, Any]]) -> None:
            for item in items:
                memory_id = str(item.get("id", ""))
                if memory_id and memory_id in dedup_ids:
                    continue
                if memory_id:
                    dedup_ids.add(memory_id)
                memory_hits.append(item)
                if len(memory_hits) >= limit:
                    return

        if self.repository and self.repository.enabled:
            _push(
                self.repository.search_preferences(
                    question=question,
                    topic_collection=active_collection,
                    limit=limit,
                )
            )
            if len(memory_hits) < limit:
                _push(
                    self.repository.search_tasks(
                        question=question,
                        topic_collection=active_collection,
                        limit=limit,
                    )
                )
            if len(memory_hits) < limit:
                _push(
                    self.repository.search_decisions(
                        question=question,
                        topic_collection=active_collection,
                        limit=limit,
                    )
                )
            if len(memory_hits) < limit:
                _push(
                    self.repository.search_episodes(
                        question=question,
                        topic_collection=active_collection,
                        limit=limit,
                        time_window_days=time_window_days,
                    )
                )
        elif self.graph_query and self.graph_query.enabled:
            _push(
                self.graph_query.retrieve_decisions(
                    question=question,
                    active_collection=active_collection,
                    limit=limit,
                )
            )

        if len(memory_hits) >= limit:
            return memory_hits[:limit]

        # Local fallback memory search for single-user mode.
        query_tokens = {token for token in re.findall(r"[A-Za-z0-9_\\-]{3,}", question.lower())}
        for source_items, memory_type in (
            (self.local_preferences, "procedural"),
            (self.local_tasks, "task"),
            (self.local_decisions, "semantic"),
            (self.local_episodes, "episodic"),
        ):
            for item in reversed(source_items):
                if active_collection and item.get("topic_collection") not in {active_collection, "", "general"}:
                    continue
                haystack = f"{item.get('title', '')} {item.get('rationale', '')}".lower()
                if any(token in haystack for token in query_tokens):
                    local_item = item.copy()
                    local_item.setdefault("memory_type", memory_type)
                    _push([local_item])
                    if len(memory_hits) >= limit:
                        return memory_hits[:limit]

        return memory_hits[:limit]

    def _extract_decision_candidate(
        self,
        question: str,
        response: str,
        active_collection: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not self.config.decision_extraction_enabled:
            return None

        question_lower = (question or "").lower()
        markers = ["decid", "aleg", "hotar", "concluz", "plan", "urmatorul pas"]
        if not any(marker in question_lower for marker in markers):
            return None

        title = (question or "").strip()
        if len(title) > 120:
            title = title[:117] + "..."
        rationale = (response or "").strip()
        if len(rationale) > 800:
            rationale = rationale[:800] + "..."

        if not title or not rationale:
            return None

        decision_id = f"decision_{hashlib.sha1(f'{title}_{rationale}'.encode('utf-8')).hexdigest()[:12]}"
        return {
            "id": decision_id,
            "title": title,
            "rationale": rationale,
            "topic_collection": active_collection or "",
            "confidence": 0.75,
            "memory_type": "semantic",
            "source": "decision_extractor",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _extract_preference_candidates(
        self,
        question: str,
        active_collection: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Extract procedural memory candidates from user intent/preferences."""
        text = (question or "").strip()
        text_lower = text.lower()
        markers = ["prefer", "vreau", "foloseste", "evita", "te rog sa", "seteaza"]
        if not any(marker in text_lower for marker in markers):
            return []

        candidates: List[Dict[str, Any]] = []
        rules = [
            ("response_style", r"(?:raspuns|explicatie)\s+(?:scurt|scurta|concis)", "Raspunsuri concise"),
            ("response_style", r"(?:raspuns|explicatie)\s+(?:detaliat|detaliata|lung)", "Raspunsuri detaliate"),
            ("citation_mode", r"(?:cu|adauga)\s+(?:surse|citari|provenienta)", "Include citari in raspuns"),
            ("language", r"\b(in|pe)\s+romana\b", "Raspunde in romana"),
            ("language", r"\b(in|pe)\s+engleza\b", "Raspunde in engleza"),
        ]

        for pref_key, pattern, pref_value in rules:
            if re.search(pattern, text_lower):
                pref_id = hashlib.sha1(
                    f"{pref_key}_{pref_value}_{active_collection or ''}".encode("utf-8")
                ).hexdigest()[:12]
                candidates.append(
                    {
                        "id": f"pref_{pref_id}",
                        "preference_key": pref_key,
                        "preference_value": pref_value,
                        "title": f"Preferinta: {pref_key}",
                        "rationale": pref_value,
                        "topic_collection": active_collection or "",
                        "confidence": 0.82,
                        "memory_type": "procedural",
                        "source": "preference_extractor",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

        return candidates

    def _persist_preferences(self, preferences: List[Dict[str, Any]]) -> None:
        if not preferences:
            return

        for preference in preferences:
            preference.setdefault("memory_type", "procedural")
            preference.setdefault("source", "preference")
            preference["updated_at"] = datetime.now(timezone.utc).isoformat()
            preference.setdefault("created_at", preference["updated_at"])

            pref_key = preference.get("preference_key", "")
            pref_topic = preference.get("topic_collection", "")
            matched = None
            for item in self.local_preferences:
                if (
                    item.get("preference_key") == pref_key
                    and item.get("topic_collection", "") == pref_topic
                ):
                    matched = item
                    break

            if matched is None:
                self.local_preferences.append(preference.copy())
            else:
                previous_conf = float(matched.get("confidence", 0.0) or 0.0)
                current_conf = float(preference.get("confidence", 0.0) or 0.0)
                # Allow override when confidence is at least as strong.
                if current_conf >= previous_conf:
                    matched.update(preference)

            if self.repository and self.repository.enabled:
                self.repository.upsert_preference(
                    preference_key=preference.get("preference_key", ""),
                    preference_value=preference.get("preference_value", ""),
                    topic_collection=preference.get("topic_collection", ""),
                    confidence=float(preference.get("confidence", 0.8)),
                )

    # Markeri care indica intent CLAR de capturare nota ("noteaza asta", "vreau sa retin").
    _NOTE_EXPLICIT_MARKERS = (
        "noteaza",
        "salveaza ca nota",
        "salveaza asta ca nota",
        "creeaza o nota",
        "creeaza nota",
        "adauga ca nota",
        "vreau sa retin asta",
        "retin asta",
        "save this as a note",
        "save as note",
        "create a note",
        "let me note this",
    )

    # Markeri care semnaleaza insight/reflectie (mai slab — propunem confirmare).
    _NOTE_INSIGHT_MARKERS = (
        "am realizat ca",
        "am realizat că",
        "am observat ca",
        "am observat că",
        "imi dau seama ca",
        "îmi dau seama că",
        "concluzia mea este",
        "concluzia mea e",
        "din experienta",
        "din experiența",
        "important de retinut",
        "important de reținut",
        "merita retinut",
        "merită reținut",
        "ma gandesc ca",
        "mă gândesc că",
        "i realized that",
        "i noticed that",
        "key takeaway",
        "worth noting",
        "interesting that",
    )

    def _extract_note_candidates(
        self,
        question: str,
        response: str,
        active_collection: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Detectează pasaje 'note-worthy' din schimbul de chat.

        Returneaza candidati cu `confidence` care decide tier-ul in query():
          - >= 0.85 → auto-save silent + badge in chat
          - 0.50-0.85 → confirm card in chat ("Vrei sa salvez asta ca nota?")
          - < 0.50 → drop
        Detectie pe markeri (cheap, fara LLM call suplimentar):
          - Markeri EXPLICITI ("noteaza", "salveaza ca nota") → conf 0.9, content = ce vine dupa marker
          - Markeri de INSIGHT ("am realizat ca", "observ ca") → conf 0.7, content = intreaga intrebare
        """
        text = (question or "").strip()
        if not text:
            return []
        text_lower = text.lower()

        # 1. Marker explicit: utilizatorul cere clar capturarea
        explicit_patterns = [
            r"(?:noteaza|salveaza\s+(?:asta\s+)?ca\s+nota|adauga\s+ca\s+nota|creeaza\s+(?:o\s+)?nota)\s*[:\-]?\s*(.+)$",
            r"(?:vreau\s+sa\s+retin|retin)\s*[:\-]?\s*(.+)$",
            r"(?:save\s+(?:this\s+)?as\s+(?:a\s+)?note|create\s+a\s+note|let\s+me\s+note\s+this)\s*[:\-]?\s*(.+)$",
        ]
        for pattern in explicit_patterns:
            match = re.search(pattern, text_lower)
            if match:
                content = match.group(1).strip(" .,:;\"'") if match.lastindex else ""
                if not content or len(content) < 5:
                    # marker singur, fara payload — folosim raspunsul ca content
                    content = (response or "").strip()
                    if len(content) > 500:
                        content = content[:500] + "..."
                if content:
                    title = content[:60].rstrip(" .,:;") + ("..." if len(content) > 60 else "")
                    return [{
                        "title": title,
                        "content": content,
                        "topic_collection": active_collection or "",
                        "confidence": 0.90,
                        "source": "extracted_explicit",
                    }]

        # 2. Marker de insight: utilizatorul exprima o reflectie → propunere
        for marker in self._NOTE_INSIGHT_MARKERS:
            if marker in text_lower:
                content = text.strip(" .,:;\"'")
                if len(content) > 500:
                    content = content[:500] + "..."
                title = content[:60].rstrip(" .,:;") + ("..." if len(content) > 60 else "")
                return [{
                    "title": title,
                    "content": content,
                    "topic_collection": active_collection or "",
                    "confidence": 0.65,
                    "source": "extracted_insight",
                }]

        return []

    def _process_note_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Aplica tier-uri pe candidatii de note.

        Returneaza {auto_saved: [...], proposed: [...]} pentru a fi propagate
        in result-ul query() si afisate in chat.
        """
        result = {"auto_saved": [], "proposed": []}
        for cand in candidates or []:
            conf = float(cand.get("confidence", 0.0) or 0.0)
            if conf >= 0.85:
                note_id = self.create_note(
                    content=cand.get("content", ""),
                    title=cand.get("title", ""),
                    topic_collection=cand.get("topic_collection", "") or "",
                )
                if note_id:
                    result["auto_saved"].append({
                        "id": note_id,
                        "title": cand.get("title", ""),
                        "content_preview": (cand.get("content", "") or "")[:120],
                        "topic_collection": cand.get("topic_collection", "") or "",
                        "confidence": conf,
                    })
            elif conf >= 0.50:
                result["proposed"].append({
                    "title": cand.get("title", ""),
                    "content": cand.get("content", ""),
                    "topic_collection": cand.get("topic_collection", "") or "",
                    "confidence": conf,
                })
        return result

    def _extract_task_candidates(
        self,
        question: str,
        response: str,
        active_collection: Optional[str],
    ) -> List[Dict[str, Any]]:
        text = (question or "").strip()
        text_lower = text.lower()
        if not text:
            return []

        # Do not create tasks for pure listing/status queries.
        if any(
            marker in text_lower
            for marker in {"ce task", "lista task", "taskurile", "task-urile", "ce am de facut"}
        ):
            return []

        markers = {
            "task",
            "todo",
            "to do",
            "reminder",
            "reaminteste",
            "adu-mi aminte",
            "trebuie sa",
            "creeaza un task",
            "adauga task",
        }
        if not any(marker in text_lower for marker in markers):
            return []

        extracted_title = ""
        patterns = [
            r"(?:creeaza|adauga)\s+(?:un\s+)?task(?:\s*[:\-])?\s*(.+)$",
            r"(?:trebuie\s+sa)\s+(.+)$",
            r"(?:adu-mi aminte\s+sa)\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                extracted_title = match.group(1).strip(" .,:;")
                break
        if not extracted_title:
            extracted_title = text.strip(" .,:;")

        if len(extracted_title) > 160:
            extracted_title = extracted_title[:157] + "..."
        if not extracted_title:
            return []

        details = (response or "").strip()
        if len(details) > 700:
            details = details[:700] + "..."

        task_id = hashlib.sha1(
            f"{extracted_title}_{active_collection or ''}".encode("utf-8")
        ).hexdigest()[:12]
        return [
            {
                "id": f"task_{task_id}",
                "title": extracted_title,
                "details": details,
                "topic_collection": active_collection or "",
                "status": "open",
                "priority": "normal",
                "confidence": 0.78,
                "memory_type": "task",
                "source": "extracted_from_chat",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ]

    def _persist_tasks(self, tasks: List[Dict[str, Any]]) -> None:
        if not tasks:
            return

        existing_ids = {item.get("id") for item in self.local_tasks}
        for task in tasks:
            task_id = task.get("id")
            if task_id and task_id not in existing_ids:
                self.local_tasks.append(task.copy())
                existing_ids.add(task_id)

            if self.repository and self.repository.enabled:
                self.repository.create_task(
                    title=task.get("title", ""),
                    details=task.get("details", ""),
                    topic_collection=task.get("topic_collection", ""),
                    status=task.get("status", "open"),
                    priority=task.get("priority", "normal"),
                    confidence=float(task.get("confidence", 0.7) or 0.7),
                )

    def list_tasks(
        self,
        active_collection: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        if self.repository and self.repository.enabled:
            return self.repository.list_tasks(
                topic_collection=active_collection,
                status=status,
                limit=limit,
            )

        scoped_tasks: List[Dict[str, Any]] = []
        status_filter = (status or "").strip().lower()
        for item in self.local_tasks:
            if active_collection and item.get("topic_collection") not in {"", "general", active_collection}:
                continue
            if status_filter and str(item.get("status", "")).lower() != status_filter:
                continue
            scoped_tasks.append(item.copy())
        scoped_tasks.sort(key=lambda value: value.get("updated_at", ""), reverse=True)
        return scoped_tasks[: max(1, int(limit))]

    def update_task_status(self, task_id: Union[str, int], status: str) -> bool:
        clean_status = (status or "").strip().lower()
        if clean_status not in {"open", "in_progress", "done", "cancelled"}:
            return False

        numeric_id: Optional[int] = None
        try:
            numeric_id = int(task_id)
        except Exception:
            numeric_id = None

        if self.repository and self.repository.enabled and numeric_id is not None:
            return self.repository.update_task_status(numeric_id, clean_status)

        for task in self.local_tasks:
            if str(task.get("id", "")) == str(task_id):
                task["status"] = clean_status
                task["updated_at"] = datetime.now(timezone.utc).isoformat()
                return True
        return False

    def _apply_memory_decay(self) -> None:
        if self.config.memory_decay_days <= 0:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(self.config.memory_decay_days))

        def _is_recent(item: Dict[str, Any]) -> bool:
            timestamp = item.get("updated_at") or item.get("created_at")
            if not timestamp:
                return True
            try:
                normalized = str(timestamp).replace("Z", "+00:00")
                dt = datetime.fromisoformat(normalized)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt >= cutoff
            except Exception:
                return True

        self.local_episodes = [item for item in self.local_episodes if _is_recent(item)]

    def _run_memory_consolidation(self, active_collection: Optional[str]) -> None:
        if not self.config.memory_consolidation_enabled:
            return

        now = datetime.now(timezone.utc)
        if self.last_memory_consolidation_at:
            try:
                last_run = datetime.fromisoformat(self.last_memory_consolidation_at.replace("Z", "+00:00"))
                if now - last_run < timedelta(minutes=5):
                    return
            except Exception:
                pass

        marker_tokens = {"am decis", "concluzie", "hotaram", "urmatorul pas", "plan"}
        existing_titles = {
            str(item.get("title", "")).strip().lower()
            for item in self.local_decisions
        }
        for episode in self.local_episodes[-40:]:
            if active_collection and episode.get("topic_collection") not in {"", "general", active_collection}:
                continue
            rationale = str(episode.get("rationale", "")).strip()
            lowered = rationale.lower()
            if not rationale or not any(token in lowered for token in marker_tokens):
                continue
            title = str(episode.get("title", "Consolidare memorie")).strip()[:120]
            if not title or title.lower() in existing_titles:
                continue
            decision = {
                "id": f"decision_{hashlib.sha1(f'consolidated_{title}'.encode('utf-8')).hexdigest()[:12]}",
                "title": title,
                "rationale": rationale[:800],
                "topic_collection": episode.get("topic_collection", "") or "",
                "confidence": max(0.55, float(episode.get("confidence", 0.6))),
                "memory_type": "semantic",
                "source": "memory_consolidation",
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
            self._persist_decision(decision)
            existing_titles.add(title.lower())

        self._apply_memory_decay()
        self.last_memory_consolidation_at = now.isoformat()

    def _capture_episode(
        self,
        question: str,
        response: str,
        active_collection: Optional[str],
    ) -> None:
        """Store lightweight episodic memory for local fallback mode."""
        question_clean = (question or "").strip()
        response_clean = (response or "").strip()
        if not question_clean or not response_clean:
            return

        episode_id = hashlib.sha1(
            f"{question_clean}_{response_clean[:200]}_{active_collection or ''}".encode("utf-8")
        ).hexdigest()[:12]
        existing_ids = {item.get("id") for item in self.local_episodes}
        if episode_id in existing_ids:
            return

        self.local_episodes.append(
            {
                "id": f"episode_{episode_id}",
                "title": question_clean[:140],
                "rationale": response_clean[:900],
                "topic_collection": active_collection or "",
                "confidence": 0.6,
                "memory_type": "episodic",
                "source": "chat_episode",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(self.local_episodes) > 200:
            self.local_episodes = self.local_episodes[-200:]

    def _persist_decision(self, decision: Dict[str, Any]) -> None:
        if not decision:
            return
        decision.setdefault("memory_type", "semantic")
        decision.setdefault("source", "decision")
        decision.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
        decision.setdefault("created_at", datetime.now(timezone.utc).isoformat())

        # Local fallback memory.
        existing_ids = {item.get("id") for item in self.local_decisions}
        if decision.get("id") not in existing_ids:
            self.local_decisions.append(decision)

        # Persist to Postgres if available.
        if self.repository and self.repository.enabled:
            self.repository.save_decision(
                title=decision.get("title", ""),
                rationale=decision.get("rationale", ""),
                topic_collection=decision.get("topic_collection", ""),
                confidence=float(decision.get("confidence", 0.75)),
            )

        # Mirror decision into graph if available.
        if self.graph_ingestion and self.graph_ingestion.enabled:
            self.graph_ingestion.ingest_decision(
                decision_id=decision.get("id", ""),
                title=decision.get("title", ""),
                rationale=decision.get("rationale", ""),
                topic_collection=decision.get("topic_collection", ""),
            )

    @staticmethod
    def _is_insufficient_response(response: str) -> bool:
        response_text = (response or "").strip().upper()
        return response_text.startswith("INSUFFICIENT_CONTEXT")

    @property
    def web_search_available(self) -> bool:
        """True daca Tavily key e configurat (din config sau env)."""
        key = (self.config.tavily_api_key or "").strip() or os.getenv("TAVILY_API_KEY", "").strip()
        return bool(key)

    def _fetch_external_sources(
        self,
        question: str,
        max_results: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch web sources via Tavily. Foloseste config (tavily_api_key, search_depth)."""
        tavily_api_key = (self.config.tavily_api_key or "").strip() or os.getenv("TAVILY_API_KEY", "").strip()
        if not tavily_api_key:
            return []

        depth = (self.config.web_fallback_search_depth or "advanced").strip().lower()
        if depth not in {"basic", "advanced"}:
            depth = "advanced"

        payload = {
            "api_key": tavily_api_key,
            "query": question,
            "max_results": int(max_results or self.config.web_fallback_max_results or 5),
            "search_depth": depth,
            "include_answer": False,
        }

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            sources = []
            for item in data.get("results", []):
                sources.append({
                    "title": item.get("title", "Sursă externă"),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                })
            return sources
        except Exception as e:
            logger.warning(f"Nu s-au putut obține surse externe: {e}")
            return []

    def _create_external_prompt(self, question: str, external_sources: List[Dict[str, Any]]) -> str:
        if not external_sources:
            return f"""Ești CerebrumAI. Nu există surse interne suficiente pentru această întrebare.

ÎNTREBARE: {question}

INSTRUCȚIUNI:
- Oferă un răspuns general bazat pe cunoștințe externe ale modelului.
- Menționează explicit că răspunsul nu este fundamentat în sursele locale.
- Formulează răspunsul în română, concis și practic.
"""

        source_blocks = []
        for idx, source in enumerate(external_sources, 1):
            title = source.get("title", f"Sursa {idx}")
            url = source.get("url", "")
            snippet = source.get("snippet", "")
            snippet_preview = snippet[:600] + "..." if len(snippet) > 600 else snippet
            source_blocks.append(
                f"[E{idx}] {title}\nURL: {url}\nRezumat: {snippet_preview}"
            )

        sources_text = "\n\n".join(source_blocks)
        return f"""Ești CerebrumAI. Sursele interne sunt insuficiente, folosește doar sursele externe de mai jos.

SURSE EXTERNE:
{sources_text}

ÎNTREBARE: {question}

INSTRUCȚIUNI:
- Răspunde strict pe baza surselor externe de mai sus.
- Marchează citările cu formatul [E1], [E2], etc.
- Dacă sursele externe sunt insuficiente, spune explicit acest lucru.
- Formulează răspunsul în română, max 4 paragrafe.
"""

    def _run_external_fallback(
        self,
        question: str,
        start_time: float,
        retrieval_mode: str,
        active_collection: Optional[str],
        reason: str,
        route_used: str = "vector",
        router_reason: str = "",
        memory_hits: Optional[List[Dict[str, Any]]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        # Daca web fallback dezactivat si nu suntem fortati explicit, returneaza un raspuns minimal.
        if not force and not self.config.enable_web_fallback:
            response_time = time.time() - start_time
            return {
                "response": (
                    "Nu am suficient context intern pentru aceasta intrebare si "
                    "fallback-ul pe web este dezactivat. Activeaza 'Cauta pe web' "
                    "in sidebar sau apasa butonul 'Cauta pe web' din chat."
                ),
                "sources": [],
                "graph_sources": [],
                "memory_hits": memory_hits or [],
                "tasks": [],
                "provenance": [],
                "external_sources": [],
                "cached": False,
                "response_time": response_time,
                "model_used": self.llm.model_name,
                "answer_origin": "external_disabled",
                "fallback_reason": reason,
                "retrieval_mode": retrieval_mode,
                "active_collection": active_collection,
                "route_used": route_used,
                "router_reason": router_reason or f"fallback_disabled:{reason}",
            }

        external_sources = self._fetch_external_sources(question)
        external_prompt = self._create_external_prompt(question, external_sources)

        self.rate_limiter.wait_if_needed()
        self.rate_limiter.add_request()
        external_response = self.llm.generate(external_prompt)
        self.stats["llm_calls"] += 1

        response_time = time.time() - start_time
        self._update_avg_response_time(response_time)

        if self.repository and self.repository.enabled:
            self.repository.log_retrieval(
                question=question,
                route_used=route_used,
                latency_ms=response_time * 1000.0,
                metrics={
                    "fallback_reason": reason,
                    "external_sources": len(external_sources),
                    "memory_hits": len(memory_hits or []),
                },
            )

        preference_candidates = self._extract_preference_candidates(
            question=question,
            active_collection=active_collection,
        )
        if preference_candidates:
            self._persist_preferences(preference_candidates)

        self._capture_episode(
            question=question,
            response=external_response,
            active_collection=active_collection,
        )

        task_candidates = self._extract_task_candidates(
            question=question,
            response=external_response,
            active_collection=active_collection,
        )
        if task_candidates:
            self._persist_tasks(task_candidates)

        self._run_memory_consolidation(active_collection=active_collection)
        tasks_snapshot = self.list_tasks(active_collection=active_collection, limit=20)

        return {
            "response": external_response,
            "sources": [],
            "graph_sources": [],
            "memory_hits": memory_hits or [],
            "tasks": tasks_snapshot,
            "provenance": [],
            "external_sources": external_sources,
            "cached": False,
            "response_time": response_time,
            "model_used": self.llm.model_name,
            "answer_origin": "external",
            "fallback_reason": reason,
            "retrieval_mode": retrieval_mode,
            "active_collection": active_collection,
            "route_used": route_used,
            "router_reason": router_reason or f"fallback:{reason}",
        }

    def query(
        self,
        question: str,
        retrieval_mode: str = "auto",
        active_collection: Optional[str] = None,
        general_collection: str = "general",
        time_window_days: Optional[int] = None,
        include_memory: bool = True,
        force_web: bool = False,
    ) -> Dict[str, Any]:
        """Procesează o întrebare și returnează răspunsul.

        force_web=True bypaseaza retrievalul intern si interogheaza direct Tavily.
        """
        start_time = time.time()
        self.stats["total_queries"] += 1

        # Web search forcat: skip retrievalul intern complet.
        if force_web:
            memory_hits = self._retrieve_memory_hits(
                question=question,
                active_collection=active_collection,
                limit=3,
                time_window_days=time_window_days,
            ) if include_memory else []
            return self._run_external_fallback(
                question,
                start_time,
                retrieval_mode,
                active_collection,
                reason="user_forced_web_search",
                route_used="external",
                router_reason="user_forced",
                memory_hits=memory_hits,
                force=True,
            )

        try:
            scope_mode = retrieval_mode if retrieval_mode in {"topic", "topic_general", "all"} else "topic_general"
            scope_filter = self._build_scope_filter(
                scope_mode,
                active_collection,
                general_collection
            )
            collection_filters = self._resolve_collection_filters(
                scope_mode,
                active_collection,
                general_collection,
            )

            # Regăsește context vectorial.
            vector_docs = self._run_vector_retrieval(
                question,
                scope_filter=scope_filter,
                collection_filters=collection_filters,
                k=self.config.max_context_docs,
            )

            # Router intent -> retrieval plan.
            vector_confidence = (
                self.hybrid_fusion.estimate_vector_confidence(vector_docs)
                if self.hybrid_fusion is not None
                else 0.0
            )

            if self.intent_router is not None:
                intent = self.intent_router.classify(question, time_window_days=time_window_days)
                plan = self.intent_router.build_plan(
                    intent=intent,
                    retrieval_mode=retrieval_mode,
                    vector_top_k=self.config.vector_top_k,
                    graph_top_k_paths=self.config.graph_top_k_paths,
                    hybrid_rerank_top_k=self.config.hybrid_rerank_top_k,
                    vector_confidence=vector_confidence,
                )
                route_used = plan.route
                router_reason = plan.reason
            else:
                route_used = "vector"
                router_reason = "query intelligence unavailable"

            graph_paths: List[Dict[str, Any]] = []
            if route_used == "hybrid" and self.config.enable_graph_rag:
                graph_paths = self._run_graph_retrieval(
                    question=question,
                    active_collection=active_collection,
                    max_paths=self.config.graph_top_k_paths,
                )

            memory_hits = self._retrieve_memory_hits(
                question=question,
                active_collection=active_collection,
                limit=5,
                time_window_days=time_window_days,
            ) if include_memory else []

            # Vector search peste notele utilizatorului (primitiva Second Brain).
            # Scope: daca esti intr-un notebook in mod "topic" strict, doar notele acelui topic.
            # Altfel (topic_general / all / Second Brain), toate notele.
            notes_topic_scope = active_collection if scope_mode == "topic" else None
            note_hits = self.search_notes(
                query=question,
                topic_collection=notes_topic_scope,
                top_k=3,
            ) if include_memory else []

            if not vector_docs and not graph_paths and not note_hits:
                return self._run_external_fallback(
                    question,
                    start_time,
                    retrieval_mode,
                    active_collection,
                    reason="no_internal_context",
                    route_used=route_used,
                    router_reason=router_reason,
                    memory_hits=memory_hits,
                )

            context_payload = self._build_route_context(
                question=question,
                route=route_used,
                vector_docs=vector_docs,
                graph_paths=graph_paths,
                memory_hits=memory_hits,
                note_hits=note_hits,
            )
            prompt = context_payload.get("prompt", "")
            selected_vector_docs = context_payload.get("selected_vector_docs", vector_docs)
            provenance = context_payload.get("provenance", [])
            graph_sources = context_payload.get("graph_sources", [])
            note_sources = context_payload.get("note_sources", [])

            if not prompt:
                return self._run_external_fallback(
                    question,
                    start_time,
                    retrieval_mode,
                    active_collection,
                    reason="context_budget_empty",
                    route_used=route_used,
                    router_reason=router_reason,
                    memory_hits=memory_hits,
                )
            
            # Verifică cache
            cached_response = None
            if self.cache_manager:
                model_settings = self.llm.get_model_info()
                cached_response = self.cache_manager.get_cached_response(prompt, model_settings)
                
                if cached_response:
                    self.stats["cache_hits"] += 1
                    if self._is_insufficient_response(cached_response):
                        return self._run_external_fallback(
                            question,
                            start_time,
                            retrieval_mode,
                            active_collection,
                            reason="insufficient_cached_context",
                            route_used=route_used,
                            router_reason=router_reason,
                            memory_hits=memory_hits,
                        )

                    response_time = time.time() - start_time
                    if self.repository and self.repository.enabled:
                        self.repository.log_retrieval(
                            question=question,
                            route_used=route_used,
                            latency_ms=response_time * 1000.0,
                            metrics={
                                "cached": True,
                                "vector_docs": len(selected_vector_docs),
                                "graph_paths": len(graph_sources),
                            },
                        )

                    preference_candidates = self._extract_preference_candidates(
                        question=question,
                        active_collection=active_collection,
                    )
                    if preference_candidates:
                        self._persist_preferences(preference_candidates)

                    task_candidates = self._extract_task_candidates(
                        question=question,
                        response=cached_response,
                        active_collection=active_collection,
                    )
                    if task_candidates:
                        self._persist_tasks(task_candidates)

                    note_candidates = self._extract_note_candidates(
                        question=question,
                        response=cached_response,
                        active_collection=active_collection,
                    )
                    note_outcome = self._process_note_candidates(note_candidates)

                    self._capture_episode(
                        question=question,
                        response=cached_response,
                        active_collection=active_collection,
                    )
                    self._run_memory_consolidation(active_collection=active_collection)

                    return {
                        "response": cached_response,
                        "sources": [doc.get("metadata", {}) for doc in selected_vector_docs],
                        "graph_sources": graph_sources,
                        "memory_hits": memory_hits,
                        "note_sources": note_sources,
                        "tasks": self.list_tasks(active_collection=active_collection, limit=20),
                        "provenance": provenance,
                        "external_sources": [],
                        "cached": True,
                        "response_time": response_time,
                        "model_used": self.llm.model_name,
                        "answer_origin": "internal",
                        "retrieval_mode": retrieval_mode,
                        "active_collection": active_collection,
                        "route_used": route_used,
                        "router_reason": router_reason,
                        "auto_captured": {
                            "tasks": list(task_candidates or []),
                            "preferences": list(preference_candidates or []),
                            "decisions": [],
                            "notes": note_outcome["auto_saved"],
                        },
                        "proposed_artifacts": {
                            "notes": note_outcome["proposed"],
                        },
                    }
            
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            self.rate_limiter.add_request()
            
            # Generează răspuns
            response = self.llm.generate(prompt)
            self.stats["llm_calls"] += 1
            
            # Salvează în cache
            if self.cache_manager and len(response) > 50:
                model_settings = self.llm.get_model_info()
                self.cache_manager.cache_response(prompt, response, model_settings)

            if self._is_insufficient_response(response):
                return self._run_external_fallback(
                    question,
                    start_time,
                    retrieval_mode,
                    active_collection,
                    reason="insufficient_internal_context",
                    route_used=route_used,
                    router_reason=router_reason,
                    memory_hits=memory_hits,
                )
            
            # Actualizează statistici
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)

            preference_candidates = self._extract_preference_candidates(
                question=question,
                active_collection=active_collection,
            )
            if preference_candidates:
                self._persist_preferences(preference_candidates)

            decision_candidate = self._extract_decision_candidate(
                question=question,
                response=response,
                active_collection=active_collection,
            )
            if decision_candidate:
                self._persist_decision(decision_candidate)

            self._capture_episode(
                question=question,
                response=response,
                active_collection=active_collection,
            )

            task_candidates = self._extract_task_candidates(
                question=question,
                response=response,
                active_collection=active_collection,
            )
            if task_candidates:
                self._persist_tasks(task_candidates)

            note_candidates = self._extract_note_candidates(
                question=question,
                response=response,
                active_collection=active_collection,
            )
            note_outcome = self._process_note_candidates(note_candidates)

            self._run_memory_consolidation(active_collection=active_collection)
            tasks_snapshot = self.list_tasks(active_collection=active_collection, limit=20)

            if self.repository and self.repository.enabled:
                self.repository.log_retrieval(
                    question=question,
                    route_used=route_used,
                    latency_ms=response_time * 1000.0,
                    metrics={
                        "cached": False,
                        "vector_docs": len(selected_vector_docs),
                        "graph_paths": len(graph_sources),
                        "memory_hits": len(memory_hits),
                    },
                )
            
            return {
                "response": response,
                "sources": [doc.get("metadata", {}) for doc in selected_vector_docs],
                "graph_sources": graph_sources,
                "memory_hits": memory_hits,
                "note_sources": note_sources,
                "tasks": tasks_snapshot,
                "provenance": provenance,
                "external_sources": [],
                "cached": False,
                "response_time": response_time,
                "model_used": self.llm.model_name,
                "context_docs_count": len(selected_vector_docs),
                "answer_origin": "internal",
                "retrieval_mode": retrieval_mode,
                "active_collection": active_collection,
                "route_used": route_used,
                "router_reason": router_reason,
                "auto_captured": {
                    "tasks": list(task_candidates or []),
                    "preferences": list(preference_candidates or []),
                    "decisions": [decision_candidate] if decision_candidate else [],
                    "notes": note_outcome["auto_saved"],
                },
                "proposed_artifacts": {
                    "notes": note_outcome["proposed"],
                },
            }
            
        except Exception as e:
            logger.error(f"Eroare la procesarea query-ului: {e}")
            return {
                "response": f"Ne pare rău, a apărut o eroare: {str(e)[:100]}...",
                "sources": [],
                "graph_sources": [],
                "memory_hits": [],
                "tasks": [],
                "provenance": [],
                "external_sources": [],
                "cached": False,
                "response_time": time.time() - start_time,
                "error": str(e),
                "answer_origin": "error",
                "route_used": "error",
                "router_reason": "exception",
            }
    
    def _update_avg_response_time(self, new_time: float):
        """Actualizează timpul mediu de răspuns"""
        if self.stats["total_queries"] == 1:
            self.stats["avg_response_time"] = new_time
        else:
            total_time = self.stats["avg_response_time"] * (self.stats["total_queries"] - 1) + new_time
            self.stats["avg_response_time"] = total_time / self.stats["total_queries"]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Returnează statusul complet al sistemului inclusiv statistici cache embeddings"""
        status = {
            "model_info": self.llm.get_model_info(),
            "stats": self.stats.copy(),
            "rate_limiter": self.rate_limiter.get_status(),
            "cache_stats": self.cache_manager.get_stats() if self.cache_manager else None,
            "embeddings_cache_stats": self.retriever.embeddings_cache.get_cache_stats() if hasattr(self.retriever, 'embeddings_cache') else None,
            "config": {
                "model_name": self.config.model_name,
                "max_context_docs": self.config.max_context_docs,
                "cache_enabled": self.config.cache_enabled,
                "chunk_size": self.config.chunk_size,
                "retrieval_candidates": self.config.retrieval_candidates,
                "rerank_top_k": self.config.rerank_top_k,
                "neighbor_window": self.config.neighbor_window,
                "hybrid_alpha": self.config.hybrid_alpha,
                "vector_top_k": self.config.vector_top_k,
                "graph_top_k_paths": self.config.graph_top_k_paths,
                "hybrid_rerank_top_k": self.config.hybrid_rerank_top_k,
                "vector_confidence_threshold": self.config.vector_confidence_threshold,
                "enable_graph_rag": self.config.enable_graph_rag,
                "context_budget_tokens": self.config.context_budget_tokens,
                "decision_extraction_enabled": self.config.decision_extraction_enabled,
                "graph_extraction_mode": self.config.graph_extraction_mode,
                "memory_consolidation_enabled": self.config.memory_consolidation_enabled,
                "memory_decay_days": self.config.memory_decay_days,
                "db_primary_strict": self.config.db_primary_strict,
                "ingestion_embedding_batch_size": self.config.ingestion_embedding_batch_size,
                "ingestion_max_workers": self.config.ingestion_max_workers,
            },
            "dependencies": {
                "genai_available": GENAI_AVAILABLE,
                "langchain_available": LANGCHAIN_AVAILABLE,
                "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
                "diskcache_available": DISKCACHE_AVAILABLE,
                "pickle_available": PICKLE_AVAILABLE,
                "storage_available": STORAGE_AVAILABLE,
                "graph_available": GRAPH_AVAILABLE,
                "query_intel_available": QUERY_INTEL_AVAILABLE,
            },
            "backends": {
                "postgres_enabled": bool(self.repository and self.repository.enabled),
                "neo4j_enabled": bool(self.graph_query and self.graph_query.enabled),
                "local_decisions_count": len(self.local_decisions),
                "local_preferences_count": len(self.local_preferences),
                "local_episodes_count": len(self.local_episodes),
                "local_tasks_count": len(self.local_tasks),
                "last_memory_consolidation_at": self.last_memory_consolidation_at,
            },
        }

        status["storage_index_status"] = self.get_storage_index_status()
        status["second_brain_status"] = self.get_second_brain_status()
        
        # Calculează cache hit rate
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
            status["cache_hit_rate"] = (self.stats["cache_hits"] / max(self.stats["total_queries"], 1)) * 100

        return status

    def get_second_brain_status(self) -> Dict[str, Any]:
        status = {
            "memory_consolidation_enabled": bool(self.config.memory_consolidation_enabled),
            "memory_decay_days": int(self.config.memory_decay_days),
            "last_memory_consolidation_at": self.last_memory_consolidation_at,
            "local": {
                "decisions": len(self.local_decisions),
                "preferences": len(self.local_preferences),
                "episodes": len(self.local_episodes),
                "tasks": len(self.local_tasks),
                "open_tasks": len(
                    [item for item in self.local_tasks if item.get("status", "open") in {"open", "in_progress"}]
                ),
            },
            "db": {
                "enabled": bool(self.repository and self.repository.enabled),
                "decisions_count": 0,
                "preferences_count": 0,
                "tasks_open_count": 0,
                "tasks_total_count": 0,
                "messages_count": 0,
            },
        }
        if self.repository and self.repository.enabled:
            db_stats = self.repository.get_second_brain_status()
            status["db"].update(db_stats)
        return status
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Returns concept/relation counts from Neo4j for dashboard metrics."""
        if self.neo4j_client and self.neo4j_client.enabled:
            return self.neo4j_client.get_graph_stats()
        return {}

    def get_graph_viz_data(
        self,
        limit_nodes: int = 150,
        relation_types=None,
        collection_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Returns nodes + edges for interactive graph visualization."""
        if self.neo4j_client and self.neo4j_client.enabled:
            return self.neo4j_client.get_graph_viz_data(
                limit_nodes=limit_nodes,
                relation_types=relation_types,
                collection_filter=collection_filter,
            )
        return {"nodes": [], "edges": []}

    def get_contradictions(self, limit: int = 20, include_dismissed: bool = False) -> List[Dict[str, Any]]:
        """Returns contradiction pairs from graph for E4 contradiction panel."""
        if self.neo4j_client and self.neo4j_client.enabled:
            return self.neo4j_client.get_contradictions(limit=limit, include_dismissed=include_dismissed)
        return []

    def list_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns most recent decisions for memory surfacing."""
        if self.repository and self.repository.enabled:
            return self.repository.list_recent_decisions(limit=limit)
        return []

    def list_aged_decisions(
        self,
        min_age_days: int = 30,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Returns decisions older than threshold (drift candidates)."""
        if self.repository and self.repository.enabled:
            return self.repository.list_aged_decisions(min_age_days=min_age_days, limit=limit)
        return []

    def list_all_decisions(self, topic_collection: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        if self.repository and self.repository.enabled:
            return self.repository.list_all_decisions(topic_collection=topic_collection, limit=limit) or []
        return []

    def update_decision(
        self,
        decision_id: int,
        title: Optional[str] = None,
        rationale: Optional[str] = None,
        topic_collection: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        if self.repository and self.repository.enabled:
            return self.repository.update_decision(
                decision_id=decision_id,
                title=title,
                rationale=rationale,
                topic_collection=topic_collection,
                confidence=confidence,
            )
        return False

    def delete_decision(self, decision_id: int) -> bool:
        if self.repository and self.repository.enabled:
            return self.repository.delete_decision(decision_id=decision_id)
        return False

    def list_all_preferences(self, topic_collection: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        if self.repository and self.repository.enabled:
            return self.repository.list_all_preferences(topic_collection=topic_collection, limit=limit) or []
        return []

    def update_preference(
        self,
        preference_id: int,
        preference_value: Optional[str] = None,
        confidence: Optional[float] = None,
        topic_collection: Optional[str] = None,
    ) -> bool:
        if self.repository and self.repository.enabled:
            return self.repository.update_preference(
                preference_id=preference_id,
                preference_value=preference_value,
                confidence=confidence,
                topic_collection=topic_collection,
            )
        return False

    def delete_preference(self, preference_id: int) -> bool:
        if self.repository and self.repository.enabled:
            return self.repository.delete_preference(preference_id=preference_id)
        return False

    def list_retrieval_logs(self, limit: int = 100, route_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.repository and self.repository.enabled:
            return self.repository.list_retrieval_logs(limit=limit, route_filter=route_filter) or []
        return []

    def get_retrieval_stats(self, days: int = 7) -> Dict[str, Any]:
        if self.repository and self.repository.enabled:
            return self.repository.get_retrieval_stats(days=days) or {}
        return {}

    def get_weekly_summary(self, days: int = 7) -> Dict[str, Any]:
        if self.repository and self.repository.enabled:
            return self.repository.get_weekly_summary(days=days) or {}
        return {}

    def dismiss_contradiction(self, source: str, target: str, note: str = "") -> bool:
        if self.neo4j_client and self.neo4j_client.enabled:
            return self.neo4j_client.dismiss_contradiction(source=source, target=target, note=note)
        return False

    def restore_contradiction(self, source: str, target: str) -> bool:
        if self.neo4j_client and self.neo4j_client.enabled:
            return self.neo4j_client.restore_contradiction(source=source, target=target)
        return False

    def list_top_preferences(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns top-confidence preferences for memory surfacing."""
        if self.repository and self.repository.enabled:
            return self.repository.list_top_preferences(limit=limit)
        return []

    # ------------------------------------------------------------------
    # Notes — user-authored Second Brain primitive
    # ------------------------------------------------------------------

    def _embed_note_text(self, title: str, content: str) -> Optional[List[float]]:
        """Compute embedding for a note (title + content)."""
        embeddings_model = getattr(self.retriever, "embeddings_model", None)
        if embeddings_model is None:
            return None
        try:
            text = f"{title}\n\n{content}".strip() if title else content.strip()
            if not text:
                return None
            vec = embeddings_model.encode([text])[0]
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            return list(vec)
        except Exception as exc:
            logger.warning("_embed_note_text failed: %s", exc)
            return None

    def create_note(
        self,
        content: str,
        title: str = "",
        topic_collection: str = "",
        tags: Optional[List[str]] = None,
    ) -> Optional[int]:
        """Create a user-authored note + compute its embedding for retrieval."""
        if not self.repository or not self.repository.enabled:
            return None
        embedding = self._embed_note_text(title, content)
        return self.repository.create_note(
            content=content,
            title=title,
            topic_collection=topic_collection,
            tags=tags,
            embedding=embedding,
        )

    def update_note(
        self,
        note_id: int,
        title: Optional[str] = None,
        content: Optional[str] = None,
        topic_collection: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Update a note. If title or content changes, recompute the embedding."""
        if not self.repository or not self.repository.enabled:
            return False
        embedding: Optional[List[float]] = None
        if title is not None or content is not None:
            current = self.repository.get_note(note_id) or {}
            final_title = title if title is not None else current.get("title", "")
            final_content = content if content is not None else current.get("content", "")
            embedding = self._embed_note_text(final_title, final_content)
        return self.repository.update_note(
            note_id=note_id,
            title=title,
            content=content,
            topic_collection=topic_collection,
            tags=tags,
            embedding=embedding,
        )

    def delete_note(self, note_id: int) -> bool:
        if not self.repository or not self.repository.enabled:
            return False
        return self.repository.delete_note(note_id)

    def get_note(self, note_id: int) -> Optional[Dict[str, Any]]:
        if not self.repository or not self.repository.enabled:
            return None
        return self.repository.get_note(note_id)

    def list_notes(
        self,
        topic_collection: Optional[str] = None,
        text_query: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not self.repository or not self.repository.enabled:
            return []
        return self.repository.list_notes(
            topic_collection=topic_collection,
            text_query=text_query,
            limit=limit,
        )

    def search_notes(
        self,
        query: str,
        topic_collection: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Vector search over notes for retrieval / surfacing."""
        if not self.repository or not self.repository.enabled:
            return []
        embedding = self._embed_note_text("", query)
        if embedding is None:
            return []
        return self.repository.vector_search_notes(
            query_embedding=embedding,
            topic_collection=topic_collection,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Synthesis on demand — agregare structurata a cunostintelor pe topic
    # ------------------------------------------------------------------

    def synthesize_topic(
        self,
        topic_collection: str,
        max_doc_chunks: int = 6,
        max_notes: int = 8,
        max_decisions: int = 5,
        max_preferences: int = 5,
    ) -> Dict[str, Any]:
        """Genereaza un outline markdown care sumarizeaza tot ce stii despre topic.

        Agregheaza:
          - chunks reprezentative din documentele topicului (vector search)
          - notele tale tagged cu topicul
          - decizii memorate in topic
          - preferinte exprimate in topic
        Apoi cere Gemini sa produca outline structurat cu citatii [D#] [N#] [Dec#] [P#].
        """
        result: Dict[str, Any] = {
            "topic": topic_collection,
            "outline": "",
            "sources_used": {"docs": [], "notes": [], "decisions": [], "preferences": []},
            "model_used": "",
            "error": None,
        }

        topic = (topic_collection or "").strip()
        if not topic:
            result["error"] = "Topic obligatoriu pentru sinteza."
            return result

        # 1. Documente: vector search cu query derivat din topic name.
        doc_chunks: List[Dict[str, Any]] = []
        if self.repository and self.repository.enabled:
            try:
                doc_query = f"concepte principale, idei centrale, concluzii, sumar despre {topic}"
                doc_embedding = self._embed_note_text("", doc_query)
                if doc_embedding:
                    doc_chunks = self.repository.vector_search(
                        query_embedding=doc_embedding,
                        collection_filters=[topic],
                        top_k=max_doc_chunks,
                    ) or []
            except Exception as exc:
                logger.warning("synthesize_topic doc retrieval failed: %s", exc)

        # 2. Note proprii.
        notes = self.list_notes(topic_collection=topic, limit=max_notes) or []

        # 3. Decizii in topic.
        decisions: List[Dict[str, Any]] = []
        if self.repository and self.repository.enabled:
            try:
                all_decisions = self.repository.list_all_decisions(limit=200) or []
                decisions = [
                    d for d in all_decisions
                    if (d.get("topic_collection") or "").strip() == topic
                ][:max_decisions]
            except Exception:
                pass

        # 4. Preferinte in topic.
        preferences: List[Dict[str, Any]] = []
        if self.repository and self.repository.enabled:
            try:
                all_prefs = self.repository.list_all_preferences(limit=200) or []
                preferences = [
                    p for p in all_prefs
                    if (p.get("topic_collection") or "").strip() == topic
                ][:max_preferences]
            except Exception:
                pass

        if not (doc_chunks or notes or decisions or preferences):
            result["error"] = (
                f"Nu ai inca nimic capturat in topicul '{topic}'. "
                "Adauga documente, note sau pune intrebari mai intai."
            )
            return result

        prompt = self._build_synthesis_prompt(topic, doc_chunks, notes, decisions, preferences)

        try:
            self.rate_limiter.wait_if_needed()
            self.rate_limiter.add_request()
            outline = self.llm.generate(prompt)
            result["outline"] = (outline or "").strip()
            result["model_used"] = self.llm.model_name
            result["sources_used"]["docs"] = [
                {
                    "filename": c.get("metadata", {}).get("filename", "?"),
                    "snippet": (c.get("content", "") or "")[:200],
                }
                for c in doc_chunks
            ]
            result["sources_used"]["notes"] = [
                {
                    "id": n.get("id"),
                    "title": n.get("title", "") or f"Nota #{n.get('id', '?')}",
                    "preview": (n.get("content", "") or "")[:200],
                }
                for n in notes
            ]
            result["sources_used"]["decisions"] = [
                {"id": d.get("id"), "title": d.get("title", "")}
                for d in decisions
            ]
            result["sources_used"]["preferences"] = [
                {
                    "id": p.get("id"),
                    "key": p.get("preference_key", ""),
                    "value": p.get("preference_value", ""),
                }
                for p in preferences
            ]
        except Exception as exc:
            logger.error("synthesize_topic generation failed: %s", exc)
            result["error"] = f"Generarea sintezei a esuat: {exc}"

        return result

    def _build_synthesis_prompt(
        self,
        topic: str,
        doc_chunks: List[Dict[str, Any]],
        notes: List[Dict[str, Any]],
        decisions: List[Dict[str, Any]],
        preferences: List[Dict[str, Any]],
    ) -> str:
        """Construieste promptul pentru sinteza topicului."""
        parts: List[str] = [
            f"Esti CerebrumAI. Sintetizeaza intr-un outline markdown structurat tot ce "
            f"stie utilizatorul despre topicul '{topic}'.",
            "",
            "MATERIAL DISPONIBIL:",
            "",
        ]

        if doc_chunks:
            parts.append("### DOCUMENTE (snippet-uri reprezentative)")
            for i, chunk in enumerate(doc_chunks, 1):
                meta = chunk.get("metadata", {}) or {}
                fname = meta.get("filename", f"doc{i}")
                content = (chunk.get("content", "") or "")[:500]
                parts.append(f"[D{i}] {fname}: {content}")
            parts.append("")

        if notes:
            parts.append("### NOTELE TALE PERSONALE (gandirea ta proprie)")
            for i, note in enumerate(notes, 1):
                title = note.get("title") or f"Nota {i}"
                content = (note.get("content", "") or "")[:500]
                parts.append(f"[N{i}] {title}: {content}")
            parts.append("")

        if decisions:
            parts.append("### DECIZII LUATE")
            for i, d in enumerate(decisions, 1):
                title = d.get("title", f"Decizie {i}")
                rationale = (d.get("rationale", "") or "")[:300]
                parts.append(f"[Dec{i}] {title}: {rationale}")
            parts.append("")

        if preferences:
            parts.append("### PREFERINTE EXPRIMATE")
            for i, p in enumerate(preferences, 1):
                key = p.get("preference_key", "")
                value = p.get("preference_value", "")
                parts.append(f"[P{i}] {key}: {value}")
            parts.append("")

        parts.extend([
            "INSTRUCTIUNI DE OUTPUT:",
            "- Genereaza markdown bine structurat, cu headings (##) si bullets.",
            "- Incepe cu o sectiune '## Privire de ansamblu' (2-3 propozitii).",
            "- Organizeaza pe sub-teme logice (descopera-le din continut, nu le forta).",
            "- Citeaza sursele: [D#] pentru documente, [N#] pentru note, [Dec#] pentru decizii, [P#] pentru preferinte.",
            "- Marcheaza explicit gap-urile observate: 'Nu ai notat inca nimic despre X' sau 'Documentele acopera Y, dar nu ai o opinie proprie notata'.",
            "- Daca exista CONTRADICTII intre note, decizii si documente, semnaleaza-le intr-o sectiune '## Tensiuni si contradictii'.",
            "- La final, sectiune '## Next steps' cu 3-5 actiuni concrete bazate pe gap-uri + decizii recente.",
            "- Limba: romana.",
            "- Lungime: 400-900 cuvinte.",
            "- Output: DOAR markdown-ul outline-ului, fara preambul.",
        ])

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Timeline — agregare cronologica a evenimentelor de cunostinte
    # ------------------------------------------------------------------

    def get_timeline_events(
        self,
        topic_collection: Optional[str] = None,
        days: int = 14,
        event_types: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Agregare cronologica a evenimentelor din cele 5 tabele de cunostinte.

        Tipuri emise:
          - note_created, note_updated
          - decision
          - task_created, task_done
          - document_added
          - chat_started

        event_types=None inseamna toate. topic_collection=None inseamna global.
        """
        if not self.repository or not self.repository.enabled:
            return []

        since = datetime.now(timezone.utc) - timedelta(days=int(days))
        since_iso = since.isoformat()
        allow = set(event_types) if event_types else None

        def _wanted(t: str) -> bool:
            return allow is None or t in allow

        def _topic_match(topic_value: Optional[str]) -> bool:
            if not topic_collection:
                return True
            return (topic_value or "").strip() == topic_collection.strip()

        events: List[Dict[str, Any]] = []

        # 1. Note (created + updated daca diferite)
        if _wanted("note_created") or _wanted("note_updated"):
            try:
                notes = self.repository.list_notes(
                    topic_collection=topic_collection,
                    limit=500,
                ) or []
                for note in notes:
                    nid = note.get("id")
                    title = note.get("title") or f"Nota #{nid}"
                    preview = (note.get("content", "") or "")[:200]
                    topic_val = note.get("topic_collection", "") or "global"
                    created = note.get("created_at", "")
                    updated = note.get("updated_at", "")
                    if _wanted("note_created") and created and created >= since_iso:
                        events.append({
                            "type": "note_created",
                            "icon": "💡",
                            "timestamp": created,
                            "topic_collection": topic_val,
                            "title": title,
                            "preview": preview,
                            "metadata": {"id": nid, "tags": note.get("tags", [])},
                        })
                    if (
                        _wanted("note_updated")
                        and updated
                        and updated >= since_iso
                        and updated != created
                    ):
                        events.append({
                            "type": "note_updated",
                            "icon": "✏️",
                            "timestamp": updated,
                            "topic_collection": topic_val,
                            "title": title,
                            "preview": preview,
                            "metadata": {"id": nid},
                        })
            except Exception as exc:
                logger.warning("timeline notes failed: %s", exc)

        # 2. Decizii
        if _wanted("decision"):
            try:
                decisions = self.repository.list_all_decisions(limit=500) or []
                for d in decisions:
                    if not _topic_match(d.get("topic_collection")):
                        continue
                    created = d.get("created_at", "")
                    if not created or created < since_iso:
                        continue
                    events.append({
                        "type": "decision",
                        "icon": "⚙️",
                        "timestamp": created,
                        "topic_collection": d.get("topic_collection", "") or "global",
                        "title": d.get("title", "Decizie"),
                        "preview": (d.get("rationale", "") or "")[:250],
                        "metadata": {
                            "id": d.get("id"),
                            "confidence": d.get("confidence", 0.0),
                        },
                    })
            except Exception as exc:
                logger.warning("timeline decisions failed: %s", exc)

        # 3. Task-uri (created + done)
        if _wanted("task_created") or _wanted("task_done"):
            try:
                tasks = self.repository.list_tasks(
                    topic_collection=topic_collection,
                    limit=500,
                ) or []
                for t in tasks:
                    tid = t.get("id")
                    title = t.get("title", f"Task #{tid}")
                    topic_val = t.get("topic_collection", "") or "global"
                    status = t.get("status", "open")
                    priority = t.get("priority", "normal")
                    created = t.get("created_at", "")
                    updated = t.get("updated_at", "")
                    if _wanted("task_created") and created and created >= since_iso:
                        events.append({
                            "type": "task_created",
                            "icon": "📌",
                            "timestamp": created,
                            "topic_collection": topic_val,
                            "title": title,
                            "preview": (t.get("details", "") or "")[:200],
                            "metadata": {"id": tid, "priority": priority, "status": status},
                        })
                    if (
                        _wanted("task_done")
                        and status == "done"
                        and updated
                        and updated >= since_iso
                        and updated != created
                    ):
                        events.append({
                            "type": "task_done",
                            "icon": "✅",
                            "timestamp": updated,
                            "topic_collection": topic_val,
                            "title": title,
                            "preview": "",
                            "metadata": {"id": tid, "priority": priority},
                        })
            except Exception as exc:
                logger.warning("timeline tasks failed: %s", exc)

        # 4. Documente
        if _wanted("document_added"):
            try:
                docs = self.repository.list_recent_documents(
                    topic_collection=topic_collection,
                    days=int(days),
                    limit=200,
                ) or []
                for doc in docs:
                    events.append({
                        "type": "document_added",
                        "icon": "📄",
                        "timestamp": doc.get("created_at", ""),
                        "topic_collection": doc.get("collection_name", "") or "global",
                        "title": doc.get("original_name", "Document"),
                        "preview": doc.get("source_path", ""),
                        "metadata": {
                            "id": doc.get("id"),
                            "indexed": doc.get("indexed", False),
                        },
                    })
            except Exception as exc:
                logger.warning("timeline documents failed: %s", exc)

        # 5. Chat sessions
        if _wanted("chat_started"):
            try:
                sessions = self.repository.list_sessions(
                    topic_collection=topic_collection,
                ) or []
                for s in sessions:
                    created = s.get("created_at", "")
                    if not created:
                        continue
                    # robust pe iso strings: filter after parse
                    if isinstance(created, str) and created < since_iso:
                        continue
                    events.append({
                        "type": "chat_started",
                        "icon": "💬",
                        "timestamp": created,
                        "topic_collection": s.get("topic_collection", "") or "global",
                        "title": s.get("title", "Chat nou"),
                        "preview": "",
                        "metadata": {
                            "id": s.get("id"),
                            "message_count": s.get("message_count", 0),
                        },
                    })
            except Exception as exc:
                logger.warning("timeline chats failed: %s", exc)

        events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return events[: int(limit)]

    # ------------------------------------------------------------------
    # Source suggestion mechanism
    # ------------------------------------------------------------------

    def suggest_sources(
        self,
        topic_collection: Optional[str] = None,
        recent_chat_history: Optional[List[Dict[str, Any]]] = None,
        max_suggestions: int = 6,
    ) -> Dict[str, Any]:
        """Suggest web sources based on conversation direction and user interests.

        Uses Tavily + Gemini to build focused search queries from recent chat
        questions, user preferences, and decisions, then returns ranked results.
        """
        result: Dict[str, Any] = {"suggestions": [], "queries_used": [], "error": None}

        if not self.web_search_available:
            result["error"] = "Tavily nu este configurat. Adauga cheia in bara laterala."
            return result

        recent_questions: List[str] = []
        if recent_chat_history:
            recent_questions = [
                e.get("question", "")
                for e in recent_chat_history[-5:]
                if e.get("question")
            ]

        preferences: List[str] = []
        decisions: List[str] = []
        if self.repository and self.repository.enabled:
            try:
                prefs = self.repository.list_top_preferences(limit=5)
                preferences = [p.get("content", "") for p in prefs if p.get("content")]
                decs = self.repository.list_recent_decisions(limit=3)
                decisions = [d.get("content", "") for d in decs if d.get("content")]
            except Exception:
                pass

        queries = self._build_suggestion_queries(
            topic_collection=topic_collection,
            recent_questions=recent_questions,
            preferences=preferences,
            decisions=decisions,
        )

        if not queries:
            result["error"] = "Nu am putut genera interogari. Incearca dupa ce ai pus cel putin o intrebare."
            return result

        seen_urls: set = set()
        per_query = max(2, (max_suggestions + len(queries) - 1) // len(queries))

        for query in queries:
            sources = self._fetch_external_sources(query, max_results=per_query + 1)
            for s in sources:
                url = s.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                result["suggestions"].append({
                    "title": s.get("title", url),
                    "url": url,
                    "snippet": s.get("snippet", ""),
                    "query_used": query,
                })
            result["queries_used"].append(query)
            if len(result["suggestions"]) >= max_suggestions:
                break

        result["suggestions"] = result["suggestions"][:max_suggestions]
        if not result["suggestions"]:
            result["error"] = "Nu s-au gasit sugestii pentru directia curenta a conversatiei."
        return result

    def _build_suggestion_queries(
        self,
        topic_collection: Optional[str],
        recent_questions: List[str],
        preferences: List[str],
        decisions: List[str],
    ) -> List[str]:
        """Use Gemini to extract 2-3 focused web search queries from conversation context."""
        if not recent_questions and not topic_collection and not preferences:
            return []

        context_parts: List[str] = []
        if topic_collection:
            context_parts.append(f"Subiect de lucru: {topic_collection}")
        if recent_questions:
            q_text = "\n".join(f"- {q}" for q in recent_questions[-4:])
            context_parts.append(f"Intrebari recente ale utilizatorului:\n{q_text}")
        if preferences:
            p_text = "\n".join(f"- {p}" for p in preferences[:3])
            context_parts.append(f"Interese / preferinte cunoscute:\n{p_text}")
        if decisions:
            d_text = "\n".join(f"- {d}" for d in decisions[:2])
            context_parts.append(f"Decizii recente:\n{d_text}")

        context_text = "\n\n".join(context_parts)

        prompt = (
            "Pe baza contextului de mai jos, genereaza 2-3 interogari de cautare web "
            "concise, in engleza, pentru a gasi resurse utile (articole, tutoriale, "
            "documentatie, papers, ghiduri) relevante pentru utilizator. "
            "Fiecare interogare sa fie specifica si diferita. "
            "Raspunde DOAR cu interogari, una pe linie, fara numerotare sau explicatii.\n\n"
            f"{context_text}"
        )

        try:
            self.rate_limiter.wait_if_needed()
            self.rate_limiter.add_request()
            raw = self.llm.generate(prompt)
            queries = [
                line.strip()
                for line in raw.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            return queries[:3]
        except Exception as exc:
            logger.warning("_build_suggestion_queries failed: %s", exc)
            return [q for q in recent_questions[-2:] if q]

    def list_all_tasks(
        self,
        topic_collection: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """List tasks cu filtre optionale (topic / status). Folosit in Tasks view."""
        if self.repository and self.repository.enabled:
            return self.repository.list_tasks(
                topic_collection=topic_collection,
                status=status,
                limit=limit,
            ) or []
        return []

    def list_due_soon_tasks(self, within_days: int = 7, limit: int = 20) -> List[Dict[str, Any]]:
        """Tasks cu due_at apropiat sau expirate (open/in_progress)."""
        if self.repository and self.repository.enabled:
            return self.repository.list_due_soon_tasks(within_days=within_days, limit=limit) or []
        return []

    def create_task_manual(
        self,
        title: str,
        details: str = "",
        topic_collection: str = "",
        priority: str = "normal",
        due_at: Optional[str] = None,
    ) -> Optional[int]:
        """Creeaza un task manual (quick capture din UI)."""
        if self.repository and self.repository.enabled:
            return self.repository.create_task(
                title=title,
                details=details,
                topic_collection=topic_collection,
                priority=priority,
                due_at=due_at,
                confidence=1.0,
            )
        return None

    def update_task(
        self,
        task_id: int,
        title: Optional[str] = None,
        details: Optional[str] = None,
        priority: Optional[str] = None,
        due_at: Optional[str] = None,
        topic_collection: Optional[str] = None,
    ) -> bool:
        if self.repository and self.repository.enabled:
            return self.repository.update_task(
                task_id=task_id,
                title=title,
                details=details,
                priority=priority,
                due_at=due_at,
                topic_collection=topic_collection,
            )
        return False

    def update_task_status(self, task_id: int, status: str) -> bool:
        if self.repository and self.repository.enabled:
            return self.repository.update_task_status(task_id=task_id, status=status)
        return False

    def delete_task(self, task_id: int) -> bool:
        if self.repository and self.repository.enabled:
            return self.repository.delete_task(task_id=task_id)
        return False

    def list_open_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns open + in_progress tasks across all topics."""
        if self.repository and self.repository.enabled:
            tasks = self.repository.list_tasks(status="open", limit=limit) or []
            in_progress = self.repository.list_tasks(status="in_progress", limit=limit) or []
            combined = tasks + in_progress
            combined.sort(key=lambda t: t.get("updated_at", ""), reverse=True)
            return combined[:limit]
        return []

    def is_reindex_recommended(self) -> bool:
        """True daca embedding model a fost schimbat fata de ultima indexare."""
        cache = getattr(self.retriever, "embeddings_cache", None)
        if not cache or not hasattr(cache, "is_reindex_pending"):
            return False
        try:
            return bool(cache.is_reindex_pending())
        except Exception:
            return False

    def reindex_all_documents(
        self,
        file_paths: List[str],
        metadata_by_path: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Sterge toate embeddings (cache + DB) si reindexeaza toate documentele.

        Apelata dupa schimbarea modelului de embedding (de ex. la trecerea la multilingv).
        Returneaza summary cu counters + duratele.
        """
        summary: Dict[str, Any] = {
            "success": False,
            "chunks_deleted": 0,
            "documents_reset": 0,
            "documents_reindexed": 0,
            "duration_s": 0.0,
        }
        start_time = time.time()

        try:
            # 1. Curata cache local.
            if hasattr(self.retriever, "embeddings_cache") and self.retriever.embeddings_cache:
                self.retriever.embeddings_cache.clear_cache()

            # 2. Reset stare retriever in-memory.
            self.retriever.documents = []
            self.retriever.document_embeddings = []
            self.retriever.chunk_lookup = {}

            # 3. Sterge embeddings si chunks din DB.
            if self.repository and self.repository.enabled:
                db_result = self.repository.clear_all_embeddings()
                summary["chunks_deleted"] = int(db_result.get("chunks_deleted", 0))
                summary["documents_reset"] = int(db_result.get("documents_reset", 0))

            # 4. Reindex cu noul model.
            if file_paths:
                loaded = self.load_documents(file_paths, metadata_by_path=metadata_by_path)
                if loaded:
                    summary["documents_reindexed"] = len(file_paths)

            # 5. Confirma reindexarea la nivel de cache signature.
            if hasattr(self.retriever, "embeddings_cache") and self.retriever.embeddings_cache:
                if hasattr(self.retriever.embeddings_cache, "mark_reindex_complete"):
                    self.retriever.embeddings_cache.mark_reindex_complete()

            summary["success"] = True
        except Exception as exc:
            logger.error("reindex_all_documents failed: %s", exc)
            summary["success"] = False
            summary["error"] = str(exc)

        summary["duration_s"] = round(time.time() - start_time, 2)
        logger.info("reindex_all_documents summary: %s", summary)
        return summary

    def clear_embeddings_cache(self):
        """Curată cache-ul de embeddings"""
        if hasattr(self.retriever, 'embeddings_cache'):
            self.retriever.embeddings_cache.clear_cache()
            logger.info("Cache embeddings curățat")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Returnează informații detaliate despre cache"""
        info = {
            "response_cache": self.cache_manager.get_stats() if self.cache_manager else None,
            "embeddings_cache": self.retriever.embeddings_cache.get_cache_stats() if hasattr(self.retriever, 'embeddings_cache') else None
        }
        return info
        
        return status
    
    def clear_cache(self):
        """Curată cache-ul sistemului"""
        if self.cache_manager:
            self.cache_manager.clear_cache()
            logger.info("Cache-ul a fost curățat")

def create_apci_system(api_key: str, config_dict: Dict[str, Any] = None) -> CerebrumAISystem:
    """Factory function pentru crearea sistemului CerebrumAI"""
    
    # Configurare implicită
    default_config = RAGConfig()
    
    # Aplică configurația personalizată
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(default_config, key):
                setattr(default_config, key, value)
    
    return CerebrumAISystem(default_config, api_key)

# Export principal
__all__ = [
    'CerebrumAISystem', 
    'RAGConfig', 
    'create_apci_system',
    'OptimizedFlashLLM',
    'GeminiRateLimiter',
    'SimpleCache'
]

