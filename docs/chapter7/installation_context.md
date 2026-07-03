# Manual de instalare — context tehnic (CerebrumAI)

> Document factual, derivat din cod. Fiecare afirmație are referința la fișierul
> care o confirmă. Infrastructura reală: Streamlit Community Cloud + Supabase
> PostgreSQL (pgvector) + Neo4j AuraDB + Gemini + Tavily.

## 1. Cerințe preliminare

| Cerință | Detaliu | Sursă |
|---|---|---|
| Python | 3.11 (imaginea devcontainer); minim recomandat 3.10+ | `.devcontainer/devcontainer.json:4` |
| Dependențe | toate în `requirements.txt` | `requirements.txt` |
| Cont Google AI / Gemini | cheie `GOOGLE_API_KEY` (obligatorie) | `src/main_flash.py:400` |
| Proiect Supabase (PostgreSQL + pgvector) | pentru Second Brain complet (opțional, dar recomandat) | `src/storage/postgres_client.py` |
| Instanță Neo4j AuraDB | pentru graful de cunoștințe (opțional) | `src/graph/neo4j_client.py` |
| Cheie Tavily | pentru web search (opțional) | `src/rag_module_flash.py:3255` |

Nu este necesar un server PostgreSQL sau Neo4j local: implementarea reală
folosește servicii găzduite (Supabase, AuraDB).

## 2. Versiunea Python

- Imaginea devcontainer fixează Python **3.11** (`.devcontainer/devcontainer.json:4`).
- Nu există `runtime.txt`, `.python-version` sau `pyproject.toml` în repository →
  versiunea exactă folosită pe Streamlit Community Cloud este `[DE CONFIRMAT DE AUTOR]`
  (vezi `open_questions.md`).

## 3. Instalarea dependențelor (local)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

Note din `requirements.txt`:
- `psycopg[binary]` + `psycopg_pool` — driver PostgreSQL (pool opțional, fallback grațios — `postgres_client.py:23-29`).
- `neo4j>=5.25.0` — driver AuraDB.
- `sentence-transformers`, `torch` — embeddings + reranker (descărcare model la prima rulare).
- `google-genai>=1.64.0` — SDK Gemini (clientul nou `genai.Client`, `rag_module_flash.py:748`).

## 4. Comanda de pornire locală

```bash
streamlit run src/main_flash.py
```

Confirmare: `.devcontainer/devcontainer.json:22` rulează exact
`streamlit run src/main_flash.py` (cu flag-urile CORS/XSRF pentru Codespaces).
`config.json` este citit din directorul curent (`src/main_flash.py:100`), deci
comanda se rulează din **rădăcina proiectului**.

## 5. Directoare și fișiere necesare

Create automat la prima rulare (nu trebuie create manual):

| Cale | Rol | Sursă |
|---|---|---|
| `./data/document_library/` | bibliotecă locală (staging) documente + `library_metadata.json` | `document_manager.py:19-32` |
| `./data/uploaded_docs/` | fișiere uploadate înainte de indexare | `_documents.py:380-381` |
| `./data/chat_sessions/` | chat-uri locale (JSON) | `chat_session_manager.py:26-32` |
| `./data/cache/`, `./data/embeddings_cache/` | cache răspunsuri / embeddings / extracție graf | `rag_module_flash.py:303,396,1428` |
| `config.json` (rădăcină) | configurația funcțională | `main_flash.py:100` |
| `.env` (rădăcină, opțional local) | variabile de mediu | `main_flash.py:37` |

> Atenție Streamlit Community Cloud: aceste directoare locale **nu sunt persistente**
> între reporniri. Vezi `streamlit_deployment.md` și `errors_and_limitations.md`.

## 6. Pașii pentru Supabase (PostgreSQL + pgvector)

1. Creează un proiect Supabase (plan Free).
2. Activează extensia `vector` (pgvector): Dashboard → Database → Extensions → `vector`.
   - Codul rulează oricum `CREATE EXTENSION IF NOT EXISTS vector;` (`schema.sql:1`), dar
     activarea din dashboard evită probleme de privilegii.
3. Obține connection string-ul (recomandat: **connection pooler**, port `6543`, mod
   tranzacție — codul setează `prepare_threshold=None` exact pentru acest caz, `postgres_client.py:76-80`).
4. Asigură-te că DSN-ul include SSL (`sslmode=require`) — Supabase impune TLS.
5. Pune DSN-ul în `POSTGRES_DSN` (env) sau `config.json → storage.postgres_dsn`.
6. La pornirea aplicației, schema și migrările se aplică **automat**
   (`repositories.py:38-43` → `initialize_schema()` + `run_migrations()`).

### Activarea pgvector

- Declarată în `src/storage/schema.sql:1`.
- Validată de `PostgresClient.test_connection()` care interoghează
  `pg_extension WHERE extname='vector'` (`postgres_client.py:139`).

### Aplicarea schemei

- Automată la startup: `initialize_schema()` rulează `schema.sql`, apoi
  `run_migrations()` aplică `src/storage/migrations/*.sql` (urmărite în tabela
  `schema_migrations`).
- Manual din UI: sidebar → 🔧 Zona tehnică → „Aplică migrațiile DB”
  (`main_flash.py:739-746`).
- Tabele create (`schema.sql`): `collections`, `documents`, `chunks`,
  `chunk_embeddings`, `chats`, `messages`, `decisions`, `preferences`, `tasks`,
  `retrieval_logs`, `notes`, `note_embeddings`, `memory_artifacts`,
  `memory_artifact_embeddings`, `memory_events`, `memory_proposals`,
  `schema_migrations`.
- Indexuri: HNSW (`vector_cosine_ops`) pe `chunk_embeddings`, `note_embeddings`,
  `memory_artifact_embeddings`; GIN full-text; btree pe topicuri/timp.
- Migrări: `001_init.sql`, `002_notes.sql`, `003_memory_artifacts.sql`,
  `004_document_file_meta.sql`.

## 7. Pașii pentru Neo4j AuraDB

1. Creează o instanță AuraDB Free.
2. Notează URI-ul: trebuie să fie `neo4j+s://...` (TLS, cerut de Aura).
3. Setează `NEO4J_URI`, `NEO4J_USER` (implicit `neo4j`), `NEO4J_PASSWORD`.
4. La pornire, dacă toate cele trei există și driverul se inițializează,
   `Neo4jClient.enabled = True` (`neo4j_client.py:27`), iar constrângerile se
   creează automat (`ensure_constraints`, apelat în `rag_module_flash.py:1490-1491`).
5. Dacă instanța e oprită/inaccesibilă, inițializarea eșuează grațios și aplicația
   continuă **vector-only** (`neo4j_client.py:33-39`).

## 8. Configurarea Gemini și Tavily

- **Gemini**: `GOOGLE_API_KEY` (obligatorie). Modelul se ia din
  `config.json → models.primary_llm` (`gemini-2.5-flash`), cu fallback
  `gemini-2.0-flash-exp`. Detecția modelului real disponibil: `_detect_best_model`
  (`rag_module_flash.py:767-803`).
- **Tavily**: `TAVILY_API_KEY` (opțională). Dacă lipsește, web search-ul e dezactivat
  (checkbox-ul „Caută pe web” nu apare) — `web_search_available` (`rag_module_flash.py:3252-3256`).

## 9. Verificarea instalării

1. Pornește `streamlit run src/main_flash.py`.
2. Dacă apare ecranul „Bun venit în CerebrumAI”, introdu cheia Gemini (`main_flash.py:404`).
3. În sidebar, „Detalii sistem” (`main_flash.py:501-513`) arată:
   - „Postgres + pgvector: conectat / indisponibil”
   - „Neo4j (graph): conectat / indisponibil”
   - „Tavily (web search): conectat / indisponibil”
4. Pastila de status: 🟢 Conectat (Postgres OK) sau 🟡 Mod local (`main_flash.py:489-492`).

## 10. Probleme frecvente (rezumat)

Detaliate în `errors_and_limitations.md`. Pe scurt:
- Cheie Gemini lipsă → ecran de onboarding, aplicația nu pornește.
- Postgres indisponibil → „Mod limitat”, Notes/Tasks/Timeline/Logs dezactivate.
- pgvector neactivat → eșuează inițializarea schemei.
- Neo4j oprit → graful dezactivat, restul funcționează.
- Tavily lipsă → fără web search.
- Pe Streamlit Cloud: fișierele locale nu persistă (vezi deployment).
</content>
