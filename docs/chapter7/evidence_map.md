# Evidence map — Capitolul 7 (CerebrumAI)

Tabel de trasabilitate: fiecare afirmație tehnică din capitol are o sursă în cod.
Nivel de certitudine: **Confirmat** (vizibil direct în cod), **Dedus** (rezultă
logic din cod/infrastructură), **De confirmat** (necesită verificare manuală a autorului).

| Afirmație pentru capitol | Fișier sursă | Clasă/funcție/cheie | Certitudine | Observații |
|---|---|---|---|---|
| Aplicația pornește prin Streamlit, fișier principal `src/main_flash.py` | `.devcontainer/devcontainer.json` | `postAttachCommand.server` | Confirmat | `streamlit run src/main_flash.py --server.enableCORS false --server.enableXsrfProtection false` |
| Punctul de intrare e `main()` în Streamlit | `src/main_flash.py:865` | `main()`, `if __name__ == "__main__"` | Confirmat | — |
| Versiune Python țintă: 3.11 | `.devcontainer/devcontainer.json:4` | `image` = `...python:1-3.11-bookworm` | Confirmat | Nu există `runtime.txt`/`.python-version` → versiunea pe Streamlit Cloud e *De confirmat* |
| Dependențele se instalează din `requirements.txt` | `requirements.txt` | întreg fișierul | Confirmat | conține `streamlit`, `google-genai`, `psycopg[binary]`, `neo4j`, `sentence-transformers` etc. |
| Configurația funcțională se încarcă din `config.json` | `src/main_flash.py:97` | `load_config()` | Confirmat | merge peste `_default_config()` |
| Secretele se citesc DOAR prin `os.getenv` (nu `st.secrets`) | `src/main_flash.py:176-181,400` | `build_apci_config_dict`, `check_api_key_setup` | Confirmat | grep global: nicio referință `st.secrets` |
| `.env` se încarcă cu `load_dotenv` la pornire | `src/main_flash.py:35-37` | `load_dotenv(... / ".env")` | Confirmat | best-effort (try/except ImportError) |
| Cheia Gemini este obligatorie; lipsa ei blochează aplicația | `src/main_flash.py:398-429,867` | `check_api_key_setup()` + `st.stop()` | Confirmat | variabila: `GOOGLE_API_KEY` |
| Conexiunea Postgres folosește un singur DSN | `src/main_flash.py:178` | `POSTGRES_DSN` / `storage.postgres_dsn` | Confirmat | `psycopg.connect(self.dsn, ...)` în `postgres_client.py:110` |
| pgvector e activat automat prin schema SQL | `src/storage/schema.sql:1` | `CREATE EXTENSION IF NOT EXISTS vector;` | Confirmat | rulat din `initialize_schema()` |
| Schema + migrările se aplică automat la pornire | `src/storage/repositories.py:38-43`, `src/storage/postgres_client.py:147-218` | `ensure_ready()` → `initialize_schema()` → `run_migrations()` | Confirmat | apelat în `CerebrumAISystem.__init__` (`rag_module_flash.py:1457-1459`) |
| Dimensiunea embeddings = 384 | `src/storage/schema.sql:33`, `config.json:43` | `VECTOR(384)`, `pgvector_embedding_dim` | Confirmat | trebuie să corespundă modelului de embeddings |
| Pooling de conexiuni opțional, compatibil Supabase pooler | `src/storage/postgres_client.py:58-86` | `_get_pool`, `prepare_threshold=None` | Confirmat | comentariu explicit despre PgBouncer/Supabase port 6543 |
| Clientul Neo4j necesită URI + user + parolă | `src/graph/neo4j_client.py:23-39` | `Neo4jClient.__init__`, `self.enabled` | Confirmat | variabile: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` |
| Graful se dezactivează grațios dacă Neo4j e indisponibil | `src/graph/neo4j_client.py:33-39`, `src/rag_module_flash.py:1493-1497` | try/except → `enabled=False` | Confirmat | aplicația continuă vector-only |
| Constrângerile Neo4j se creează automat | `src/graph/neo4j_client.py:45-64`, `src/rag_module_flash.py:1490-1491` | `ensure_constraints()` | Confirmat | apelat dacă `neo4j_client.enabled` |
| Web search (Tavily) e opțional | `src/rag_module_flash.py:3252-3256` | `web_search_available` | Confirmat | True doar dacă `TAVILY_API_KEY` există |
| Modelul LLM e Gemini, cu autodetecție și fallback | `src/rag_module_flash.py:767-803`, `config.json:7-8` | `_detect_best_model`, `primary_llm`/`fallback_llm` | Confirmat | primary `gemini-2.5-flash`, fallback `gemini-2.0-flash-exp` |
| Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (384d) | `config.json:10`, `src/rag_module_flash.py:976-979` | `embedding_model`, `SentenceTransformer(...)` | Confirmat | multilingv RO↔EN |
| Reranker cross-encoder, lazy load | `config.json:52`, `src/rag_module_flash.py:1436-1444` | `reranker_model`, `MultilingualReranker` | Confirmat | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Formate acceptate la upload: TXT, MD, PDF | `src/views/notebooks.py:472-475` | `st.file_uploader(type=["txt","md","pdf"])` | Confirmat | procesorul acceptă aceleași tipuri (`rag_module_flash.py:869-884`) |
| Postgres e sursa de adevăr pentru surse/notebook-uri | `src/views/_collections.py:1-73` | `get_notebook_collections`, `get_notebook_documents` | Confirmat | biblioteca locală = staging la upload |
| Biblioteca locală scrie pe filesystem | `src/document_manager.py:19-32,107-139` | `DocumentManager`, `shutil.copy2`, `_save_metadata` | Confirmat | `./data/document_library` |
| Chat-urile locale se salvează ca JSON | `src/chat_session_manager.py:2-34` | `ChatSessionManager` | Confirmat | `./data/chat_sessions`; pot fi migrate în Postgres |
| Ingestia e sincronă (blochează UI cu spinner) | `src/views/_documents.py:480-487`, `notebooks.py:480-487` | `process_uploaded_files` în `st.spinner` | Confirmat | fără cozi/background |
| Deduplicare documente prin hash MD5 | `src/document_manager.py:88-97` | `add_document` (file_hash) | Confirmat | duplicatul returnează id-ul existent |
| Pagini aplicație (7 tab-uri) | `src/main_flash.py:835-850` | `st.navigation([...])` | Confirmat | Second Brain, Notebooks, Notes, Tasks, Timeline, Graph, Logs |
| Sursă web persistentă (salvată ca document) | `src/views/_documents.py:119-192` | `add_web_url_as_document` | Confirmat | fetch + strip HTML → `web_*.txt` indexat |
| Sursă web temporară (Tavily la query) | `src/rag_module_flash.py:3258-3274` | `_fetch_external_sources` | Confirmat | nu se persistă ca sursă |
| Notes/Tasks/Timeline/Logs necesită Postgres | `src/views/notes.py:39-42`, `tasks.py:38-41`, `retrieval_logs.py:30-33` | gard `repo.enabled` | Confirmat | altfel afișează avertisment |
| Există `neo4j_uri` specific proiectului în `config.json` | `config.json:46` | `graph.neo4j_uri` | Confirmat | de mutat în env / golit înainte de publicare (vezi open_questions) |
</content>
</invoke>
