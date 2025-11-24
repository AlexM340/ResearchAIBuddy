"""
Modulul RAG principal pentru APCI (Asistentul Personalizat de Cercetare și Învățare)
Implementează sistemul RAG avansat cu Gemini 2.5 Flash
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Google Generative AI nu este disponibil")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
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
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_enabled: bool = True
    rate_limit_rpm: int = 30
    rate_limit_rpd: int = 3000

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
    """Cache persistent pentru embeddings cu verificare hash MD5"""
    
    def __init__(self, cache_dir: str = "./data/embeddings_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Fișiere pentru cache
        self.metadata_file = self.cache_dir / "metadata.json"
        self.embeddings_file = self.cache_dir / "embeddings.pkl"
        self.documents_file = self.cache_dir / "documents.pkl"
        
        # Încarcă cache-ul existent
        self.metadata = self._load_metadata()
        self.embeddings_cache = {}
        self.documents_cache = {}
        
        logger.info(f"Cache persistent inițializat în: {self.cache_dir}")
    
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
        cached_embeddings = []
        new_files = []
        
        for file_path in file_paths:
            current_hash = self._calculate_file_hash(file_path)
            file_key = str(Path(file_path).resolve())
            
            # Verifică dacă fișierul este în cache și hash-ul se potrivește
            if (file_key in self.metadata and 
                self.metadata[file_key].get('hash') == current_hash and
                file_key in self.documents_cache and
                file_key in self.embeddings_cache):
                
                cached_documents.extend(self.documents_cache[file_key])
                if cached_embeddings == []:
                    cached_embeddings = self.embeddings_cache[file_key]
                else:
                    if NUMPY_AVAILABLE:
                        cached_embeddings = np.vstack([cached_embeddings, self.embeddings_cache[file_key]])
                    else:
                        cached_embeddings.extend(self.embeddings_cache[file_key])
                        
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
            file_docs = [doc for doc in documents if doc.get('source', '') == file_path]
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

class OptimizedFlashLLM:
    """LLM optimizat pentru Gemini 2.5 Flash"""
    
    def __init__(self, config: RAGConfig, api_key: str):
        if not GENAI_AVAILABLE:
            raise ImportError("Google Generative AI nu este disponibil")
            
        self.config = config
        self.api_key = api_key
        
        # Configurare Gemini
        genai.configure(api_key=api_key)
        
        # Detectează modelul disponibil
        self.model_name = self._detect_best_model()
        
        # Configurare generare
        self.generation_config = genai.types.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
        )
        
        # Inițializare model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config
        )
        
        logger.info(f"OptimizedFlashLLM inițializat cu {self.model_name}")
    
    def _detect_best_model(self) -> str:
        """Detectează cel mai bun model disponibil"""
        try:
            available_models = [m.name for m in genai.list_models()]
            
            # Prioritatea modelelor
            preferred_models = [
                "models/gemini-2.5-flash",
                "models/gemini-2.5-flash-preview-05-20", 
                "models/gemini-2.0-flash-exp",
                "models/gemini-2.0-flash",
                "models/gemini-1.5-flash"
            ]
            
            for model in preferred_models:
                if model in available_models:
                    logger.info(f"Model detectat: {model}")
                    return model
            
            # Fallback la primul model Flash disponibil
            flash_models = [m for m in available_models if "flash" in m.lower()]
            if flash_models:
                logger.warning(f"Folosesc fallback model: {flash_models[0]}")
                return flash_models[0]
            
            raise Exception("Nu s-a găsit niciun model Flash disponibil")
            
        except Exception as e:
            logger.error(f"Eroare la detectarea modelului: {e}")
            return self.config.fallback_model
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generează răspuns cu retry logic"""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                
                if response.text:
                    return response.text.strip()
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
    
    def load_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Încarcă documente din fișiere"""
        documents = []
        
        for file_path in file_paths:
            file_path = Path(file_path)
            
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
                    text_content = "\n".join([doc.page_content for doc in docs])
                    
                else:
                    logger.warning(f"Tip de fișier nesuportat: {file_path}")
                    continue
                
                if text_content.strip():
                    doc = {
                        'content': text_content,
                        'metadata': {
                            'filename': file_path.name,
                            'file_type': file_path.suffix,
                            'source_path': str(file_path)
                        }
                    }
                    documents.append(doc)
                    logger.info(f"Încărcat {file_path.name}")
                
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
                
                for chunk in doc_chunks:
                    if len(chunk.page_content.strip()) > 50:
                        chunks.append({
                            'content': chunk.page_content,
                            'metadata': chunk.metadata
                        })
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
        
        # Inițializare cache persistent
        self.embeddings_cache = PersistentEmbeddingsCache()
        
        # Inițializare model embeddings dacă e disponibil
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embeddings_model = SentenceTransformer(config.embedding_model)
                logger.info(f"Model embeddings încărcat: {config.embedding_model}")
            except Exception as e:
                logger.warning(f"Nu s-a putut încărca modelul embeddings: {e}")
    
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
                    self.embeddings_cache.save_embeddings(file_paths, documents, self.document_embeddings)
                    
                except Exception as e:
                    logger.error(f"Eroare la generarea embeddings: {e}")
                    self.document_embeddings = []
        else:
            logger.info("Toate documentele sunt deja în index")
        
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
        
        logger.info(f"Index actualizat cu {len(self.documents)} documente totale")
    
    def retrieve(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Regăsește documente relevante"""
        if not self.documents:
            logger.warning("Nu există documente indexate")
            return []
        
        k = k or self.config.max_context_docs
        
        if self.embeddings_model and len(self.document_embeddings) > 0:
            # Căutare semantică cu embeddings
            return self._semantic_search(query, k)
        else:
            # Căutare simplă pe bază de cuvinte cheie
            return self._keyword_search(query, k)
    
    def _semantic_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Căutare semantică cu embeddings"""
        try:
            # Generează embedding pentru query
            query_embedding = self.embeddings_model.encode([query])
            
            # Calculează similaritatea
            if NUMPY_AVAILABLE:
                similarities = np.dot(self.document_embeddings, query_embedding.T).flatten()
                top_indices = np.argsort(similarities)[::-1][:k]
            else:
                # Fallback fără numpy
                similarities = []
                for doc_emb in self.document_embeddings:
                    sim = sum(a * b for a, b in zip(doc_emb, query_embedding[0]))
                    similarities.append(sim)
                
                # Sortează și ia top k
                indexed_sims = list(enumerate(similarities))
                indexed_sims.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in indexed_sims[:k]]
            
            return [self.documents[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"Eroare la căutarea semantică: {e}")
            return self._keyword_search(query, k)
    
    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Căutare simplă pe bază de cuvinte cheie"""
        query_words = set(query.lower().split())
        
        scored_docs = []
        for doc in self.documents:
            content_words = set(doc['content'].lower().split())
            
            # Scor bazat pe intersecția cuvintelor
            word_overlap = len(query_words.intersection(content_words))
            length_score = min(len(doc['content']) / 1000, 1.0)
            
            total_score = word_overlap + length_score
            scored_docs.append((total_score, doc))
        
        # Sortează după scor
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in scored_docs[:k]]

class APCISystem:
    """Sistemul principal APCI cu toate optimizările"""
    
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
        
        # Statistici
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "avg_response_time": 0,
            "documents_indexed": 0
        }
        
        logger.info("APCISystem inițializat cu succes")
    
    def load_documents(self, file_paths: List[str]) -> bool:
        """Încarcă și indexează documente cu cache persistent pentru viteză optimă"""
        try:
            start_time = time.time()
            
            # Încarcă documente
            documents = self.document_processor.load_documents(file_paths)
            if not documents:
                logger.warning("Nu s-au încărcat documente")
                return False
            
            # Procesează documente
            processed_docs = self.document_processor.process_documents(documents)
            if not processed_docs:
                logger.warning("Nu s-au procesat documente")
                return False
            
            # Construiește index cu cache persistent - aceasta este optimizarea cheie!
            self.retriever.build_index_with_cache(file_paths, processed_docs)
            
            self.stats["documents_indexed"] = len(processed_docs)
            
            end_time = time.time()
            load_time = end_time - start_time
            
            logger.info(f"Încărcare completă în {load_time:.2f}s: {len(processed_docs)} chunks din {len(file_paths)} fișiere")
            
            return True
            
        except Exception as e:
            logger.error(f"Eroare la încărcarea documentelor: {e}")
            return False
    
    def load_documents_legacy(self, file_paths: List[str]) -> bool:
        """Încarcă și indexează documente - versiunea originală fără cache"""
        try:
            # Încarcă documente
            documents = self.document_processor.load_documents(file_paths)
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
            return f"""Ești APCI (Asistentul Personalizat de Cercetare și Învățare), un AI expert în educație și cercetare.

Întrebare: {query}

Răspunde concis și informativ, oferind informații relevante și practice. Limitează răspunsul la maximum 3 paragrafe.

Răspuns:"""
        
        # Creează context compact
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            filename = doc['metadata'].get('filename', f'Doc{i}')
            # Limitează fiecare document pentru eficiență
            content = doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
            context_parts.append(f"[{i}. {filename}]: {content}")
        
        context = "\n".join(context_parts)
        
        return f"""Ești APCI (Asistentul Personalizat de Cercetare și Învățare). Analizează contextul și răspunde la întrebare.

CONTEXT:
{context}

ÎNTREBARE: {query}

INSTRUCȚIUNI:
- Răspunde pe baza contextului furnizat
- Citează sursele relevante [număr]
- Fii concis dar complet (max 4 paragrafe)
- Dacă contextul e insuficient, menționează acest lucru

RĂSPUNS:"""
    
    def query(self, question: str) -> Dict[str, Any]:
        """Procesează o întrebare și returnează răspunsul"""
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        try:
            # Regăsește context relevant
            context_docs = self.retriever.retrieve(question, k=self.config.max_context_docs)
            
            # Creează prompt
            prompt = self._create_rag_prompt(question, context_docs)
            
            # Verifică cache
            cached_response = None
            if self.cache_manager:
                model_settings = self.llm.get_model_info()
                cached_response = self.cache_manager.get_cached_response(prompt, model_settings)
                
                if cached_response:
                    self.stats["cache_hits"] += 1
                    return {
                        "response": cached_response,
                        "sources": [doc['metadata'] for doc in context_docs],
                        "cached": True,
                        "response_time": time.time() - start_time,
                        "model_used": self.llm.model_name
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
            
            # Actualizează statistici
            response_time = time.time() - start_time
            self._update_avg_response_time(response_time)
            
            return {
                "response": response,
                "sources": [doc['metadata'] for doc in context_docs],
                "cached": False,
                "response_time": response_time,
                "model_used": self.llm.model_name,
                "context_docs_count": len(context_docs)
            }
            
        except Exception as e:
            logger.error(f"Eroare la procesarea query-ului: {e}")
            return {
                "response": f"Ne pare rău, a apărut o eroare: {str(e)[:100]}...",
                "sources": [],
                "cached": False,
                "response_time": time.time() - start_time,
                "error": str(e)
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
                "chunk_size": self.config.chunk_size
            },
            "dependencies": {
                "genai_available": GENAI_AVAILABLE,
                "langchain_available": LANGCHAIN_AVAILABLE,
                "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
                "diskcache_available": DISKCACHE_AVAILABLE,
                "pickle_available": PICKLE_AVAILABLE
            }
        }
        
        # Calculează cache hit rate
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
            status["cache_hit_rate"] = (self.stats["cache_hits"] / max(self.stats["total_queries"], 1)) * 100
        
        return status
    
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

def create_apci_system(api_key: str, config_dict: Dict[str, Any] = None) -> APCISystem:
    """Factory function pentru crearea sistemului APCI"""
    
    # Configurare implicită
    default_config = RAGConfig()
    
    # Aplică configurația personalizată
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(default_config, key):
                setattr(default_config, key, value)
    
    return APCISystem(default_config, api_key)

# Export principal
__all__ = [
    'APCISystem', 
    'RAGConfig', 
    'create_apci_system',
    'OptimizedFlashLLM',
    'GeminiRateLimiter',
    'SimpleCache'
]
