"""
Manager pentru documentele bibliotecii CerebrumAI
Gestionează persistența și managementul documentelor încărcate
"""

import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DocumentManager:
    """Manager pentru bibliotecă de documente persistente"""
    
    def __init__(self, library_path: str = "./data/document_library"):
        self.library_path = Path(library_path)
        self.docs_path = self.library_path / "documents"
        self.metadata_file = self.library_path / "library_metadata.json"
        
        # Creează directoarele necesare
        self.library_path.mkdir(parents=True, exist_ok=True)
        self.docs_path.mkdir(parents=True, exist_ok=True)
        
        # Încarcă metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Încarcă metadata bibliotecii"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Eroare la încărcarea metadata: {e}")
        
        # Metadata implicită
        return {
            "documents": {},
            "collections": {},
            "last_updated": None,
            "version": "1.0"
        }
    
    def _save_metadata(self):
        """Salvează metadata bibliotecii"""
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Eroare la salvarea metadata: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculează hash-ul unui fișier pentru identificare unică"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Eroare la calcularea hash-ului pentru {file_path}: {e}")
            return None
    
    def add_document(self, file_path: str, collection: str = "default", 
                    description: str = "") -> Optional[str]:
        """
        Adaugă un document în bibliotecă
        
        Args:
            file_path: Calea către fișierul de adăugat
            collection: Colecția în care să adauge documentul
            description: Descrierea documentului
            
        Returns:
            ID-ul documentului adăugat sau None dacă a eșuat
        """
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                logger.error(f"Fișierul nu există: {file_path}")
                return None
            
            # Calculează hash-ul pentru verificarea duplicatelor
            file_hash = self._calculate_file_hash(source_path)
            if not file_hash:
                return None
            
            # Verifică dacă documentul există deja
            for doc_id, doc_info in self.metadata["documents"].items():
                if doc_info.get("file_hash") == file_hash:
                    logger.info(f"Documentul {source_path.name} există deja în bibliotecă")
                    return doc_id
            
            # Generează ID unic
            doc_id = f"doc_{file_hash[:12]}"
            
            # Determină numele fișierului în bibliotecă
            file_extension = source_path.suffix
            library_filename = f"{doc_id}{file_extension}"
            library_file_path = self.docs_path / library_filename
            
            # Copiază fișierul în bibliotecă
            shutil.copy2(source_path, library_file_path)
            
            # Adaugă metadata
            doc_metadata = {
                "id": doc_id,
                "original_name": source_path.name,
                "library_path": str(library_file_path.relative_to(self.library_path)),
                "file_hash": file_hash,
                "file_size": source_path.stat().st_size,
                "file_type": file_extension.lower(),
                "collection": collection,
                "description": description,
                "added_date": datetime.now().isoformat(),
                "indexed": False,
                "index_date": None
            }
            
            self.metadata["documents"][doc_id] = doc_metadata
            
            # Adaugă la colecție
            if collection not in self.metadata["collections"]:
                self.metadata["collections"][collection] = {
                    "name": collection,
                    "documents": [],
                    "created_date": datetime.now().isoformat()
                }
            
            if doc_id not in self.metadata["collections"][collection]["documents"]:
                self.metadata["collections"][collection]["documents"].append(doc_id)
            
            # Salvează metadata
            self._save_metadata()
            
            logger.info(f"Document adăugat în bibliotecă: {source_path.name} -> {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Eroare la adăugarea documentului {file_path}: {e}")
            return None
    
    def remove_document(self, doc_id: str) -> bool:
        """Elimină un document din bibliotecă"""
        try:
            if doc_id not in self.metadata["documents"]:
                logger.error(f"Documentul {doc_id} nu există în bibliotecă")
                return False
            
            doc_info = self.metadata["documents"][doc_id]
            
            # Șterge fișierul
            library_file_path = self.library_path / doc_info["library_path"]
            if library_file_path.exists():
                library_file_path.unlink()
            
            # Elimină din colecții
            collection = doc_info.get("collection", "default")
            if collection in self.metadata["collections"]:
                if doc_id in self.metadata["collections"][collection]["documents"]:
                    self.metadata["collections"][collection]["documents"].remove(doc_id)
                
                # Șterge colecția dacă e goală
                if not self.metadata["collections"][collection]["documents"]:
                    del self.metadata["collections"][collection]
            
            # Elimină metadata
            del self.metadata["documents"][doc_id]
            
            # Salvează
            self._save_metadata()
            
            logger.info(f"Document eliminat din bibliotecă: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Eroare la eliminarea documentului {doc_id}: {e}")
            return False
    
    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Obține informații despre un document"""
        return self.metadata["documents"].get(doc_id)
    
    def get_all_documents(self) -> Dict[str, Dict[str, Any]]:
        """Obține toate documentele din bibliotecă"""
        return self.metadata["documents"]
    
    def get_documents_by_collection(self, collection: str) -> List[Dict[str, Any]]:
        """Obține documentele dintr-o colecție"""
        if collection not in self.metadata["collections"]:
            return []
        
        doc_ids = self.metadata["collections"][collection]["documents"]
        return [self.metadata["documents"][doc_id] for doc_id in doc_ids 
                if doc_id in self.metadata["documents"]]
    
    def get_collections(self) -> Dict[str, Dict[str, Any]]:
        """Obține toate colecțiile"""
        return self.metadata["collections"]
    
    def create_collection(self, name: str) -> bool:
        """Creează o colecție nouă"""
        if name in self.metadata["collections"]:
            return False
        
        self.metadata["collections"][name] = {
            "name": name,
            "documents": [],
            "created_date": datetime.now().isoformat()
        }
        
        self._save_metadata()
        return True
    
    def delete_collection(self, name: str) -> bool:
        """Sterge o colectie si toate documentele ei."""
        try:
            if name not in self.metadata["collections"]:
                return False
            doc_ids = list(self.metadata["collections"][name].get("documents", []))
            for doc_id in doc_ids:
                doc_info = self.metadata["documents"].get(doc_id)
                if doc_info:
                    library_file_path = self.library_path / doc_info["library_path"]
                    if library_file_path.exists():
                        library_file_path.unlink()
                    del self.metadata["documents"][doc_id]
            del self.metadata["collections"][name]
            self._save_metadata()
            logger.info(f"Colectie stearsa: {name} ({len(doc_ids)} documente)")
            return True
        except Exception as e:
            logger.error(f"Eroare la stergerea colectiei {name}: {e}")
            return False

    def move_document_to_collection(self, doc_id: str, new_collection: str) -> bool:
        """Mută un document într-o colecție nouă"""
        try:
            if doc_id not in self.metadata["documents"]:
                return False
            
            doc_info = self.metadata["documents"][doc_id]
            old_collection = doc_info.get("collection", "default")
            
            # Elimină din colecția veche
            if old_collection in self.metadata["collections"]:
                if doc_id in self.metadata["collections"][old_collection]["documents"]:
                    self.metadata["collections"][old_collection]["documents"].remove(doc_id)
            
            # Adaugă în colecția nouă
            if new_collection not in self.metadata["collections"]:
                self.create_collection(new_collection)
            
            self.metadata["collections"][new_collection]["documents"].append(doc_id)
            self.metadata["documents"][doc_id]["collection"] = new_collection
            
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"Eroare la mutarea documentului {doc_id}: {e}")
            return False
    
    def get_document_file_path(self, doc_id: str) -> Optional[Path]:
        """Obține calea către fișierul unui document"""
        if doc_id not in self.metadata["documents"]:
            return None
        
        doc_info = self.metadata["documents"][doc_id]
        library_file_path = self.library_path / doc_info["library_path"]
        
        if library_file_path.exists():
            return library_file_path
        
        return None
    
    def mark_document_indexed(self, doc_id: str):
        """Marchează un document ca fiind indexat"""
        if doc_id in self.metadata["documents"]:
            self.metadata["documents"][doc_id]["indexed"] = True
            self.metadata["documents"][doc_id]["index_date"] = datetime.now().isoformat()
            self._save_metadata()
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Obține statistici despre bibliotecă"""
        docs = self.metadata["documents"]
        
        stats = {
            "total_documents": len(docs),
            "total_collections": len(self.metadata["collections"]),
            "indexed_documents": sum(1 for doc in docs.values() if doc.get("indexed", False)),
            "total_size_bytes": sum(doc.get("file_size", 0) for doc in docs.values()),
            "file_types": {},
            "documents_by_collection": {}
        }
        
        # Statistici pe tipuri de fișiere
        for doc in docs.values():
            file_type = doc.get("file_type", "unknown")
            stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
        
        # Documente pe colecții
        for collection_name, collection_info in self.metadata["collections"].items():
            stats["documents_by_collection"][collection_name] = len(collection_info["documents"])
        
        return stats
    
    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Caută documente după nume sau descriere"""
        query_lower = query.lower()
        results = []
        
        for doc_id, doc_info in self.metadata["documents"].items():
            # Caută în nume
            if query_lower in doc_info.get("original_name", "").lower():
                results.append(doc_info)
                continue
            
            # Caută în descriere
            if query_lower in doc_info.get("description", "").lower():
                results.append(doc_info)
                continue
        
        return results

if __name__ == "__main__":
    # Test simplu
    print("🧪 Testez Document Manager...")
    dm = DocumentManager()
    print(f"✅ Document Manager inițializat: {dm.library_path}")
    
    stats = dm.get_library_stats()
    print(f"📊 Statistici: {stats}")
    print("✅ Test complet!")
