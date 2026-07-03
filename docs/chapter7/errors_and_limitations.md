# Erori și limitări practice (CerebrumAI)

Clasificare:
- **[COD]** — comportament confirmat direct în cod;
- **[INFRA]** — limitare probabilă a infrastructurii (planuri Free), dedusă;
- **[DE CONFIRMAT]** — necesită verificare manuală a autorului.

## 1. Chei / configurare lipsă

| Situație | Comportament | Clasă | Sursă |
|---|---|---|---|
| `GOOGLE_API_KEY` lipsă | Ecran de onboarding; `st.stop()` — aplicația nu pornește | [COD] | `main_flash.py:398-429,867-868` |
| `TAVILY_API_KEY` lipsă | Web search dezactivat (checkbox ascuns); sugestiile de surse nu funcționează | [COD] | `rag_module_flash.py:3252-3256` |
| `POSTGRES_DSN` lipsă | „Mod local (fără Postgres)”; Notes/Tasks/Timeline/Logs + memoria persistentă indisponibile | [COD] | `main_flash.py:489-496`, `postgres_client.py:45` |
| `NEO4J_*` lipsă | Graful dezactivat; aplicația rulează vector-only | [COD] | `neo4j_client.py:27` |

## 2. Baze de date

| Situație | Comportament | Clasă | Sursă |
|---|---|---|---|
| Conexiune Supabase eșuată | `test_connection()` returnează False; repository dezactivat; fallback local | [COD] | `postgres_client.py:127-145` |
| Extensia pgvector neactivată | `initialize_schema()` eșuează la `CREATE EXTENSION` dacă rolul nu are privilegii → storage Postgres efectiv indisponibil | [COD]/[INFRA] | `schema.sql:1`, `postgres_client.py:147-168` |
| Migrări DB în așteptare | Avertisment în sidebar + buton „Aplică migrațiile DB” | [COD] | `main_flash.py:497-498,739-746` |
| Conexiune AuraDB eșuată / instanță oprită | Driver init în try/except → `enabled=False`, graf dezactivat, restul funcționează | [COD] | `neo4j_client.py:33-39` |
| Supabase Free în pauză (după inactivitate) | Conexiunile eșuează până la reactivare → mod local temporar | [INFRA] | plan Free Supabase |
| AuraDB Free în pauză / ștearsă după inactivitate prelungită | Graful indisponibil; necesită reactivare/recreare | [INFRA] | plan Free AuraDB |
| Limite dimensiune (Supabase ~500MB / AuraDB noduri-relații) | Ingestia poate eșua la atingerea limitelor | [INFRA] | planuri Free |

## 3. LLM (Gemini)

| Situație | Comportament | Clasă | Sursă |
|---|---|---|---|
| Eroare / răspuns gol | Retry cu exponential backoff, 3 tentative; apoi excepție | [COD] | `rag_module_flash.py:805-826` |
| Rate limit | Limitator intern (rpm/rpd din `config.json`: 300/10000) | [COD] | `rag_module_flash.py:1415-1418`, `config.json:16-19` |
| Timeout / depășire rate la furnizor | Tratat ca eroare de generare (retry, apoi propagare) | [COD]/[DE CONFIRMAT] | comportamentul exact al API-ului — de observat |
| Context insuficient | Răspuns `INSUFFICIENT_CONTEXT` | [COD] | `rag_module_flash.py:3247-3250` |

## 4. Documente / ingestie

| Situație | Comportament | Clasă | Sursă |
|---|---|---|---|
| Document duplicat | Deduplicare prin hash MD5; documentul existent e refolosit, nu re-ingerat | [COD] | `document_manager.py:88-97`, `_documents.py:298-305` |
| Format nesuportat | Uploader-ul acceptă doar TXT/MD/PDF; alt tip → „Tip de fișier nesuportat”, ignorat | [COD] | `notebooks.py:472-475`, `rag_module_flash.py:882-884` |
| PDF fără text extractibil | Conținut gol după extracție → document sărit (fără chunks) | [COD] | `rag_module_flash.py:877-886` (gard `text_content.strip()`) |
| Ingestie sincronă | Procesarea rulează în request, blochează UI cu spinner; fără cozi/background | [COD] | `_documents.py:480-487` |
| Fișiere mici (<50 caractere / chunk) | Chunk-urile foarte scurte sunt eliminate | [COD] | `rag_module_flash.py:925,945` |

## 5. Streamlit Community Cloud

| Situație | Comportament | Clasă | Sursă |
|---|---|---|---|
| Filesystem ne-persistent | Fișierele brute uploadate, chat-urile JSON locale și cache-urile se pierd la repornire | [COD]/[INFRA] | `document_manager.py:19-32`, `chat_session_manager.py:26` |
| Conținut indexat după repornire | Persistă în Supabase (chunks+embeddings) dacă Postgres e configurat → retrieval funcționează | [COD] | `_collections.py:1-73` |
| „Reindexează toate documentele” după repornire | Poate eșua: necesită fișierele locale brute, care nu mai există | [COD]/[INFRA] | `_documents.py:71-116` |
| Chat-uri locale ne-migrate | Se pierd dacă nu sunt migrate în Postgres | [COD] | `main_flash.py:319-390` |
| `.env` inexistent pe Cloud | Secretele trebuie introduse în dashboard (env vars), nu prin onboarding | [COD] | `main_flash.py:35-37,416-424` |

## 6. De confirmat de autor

- Versiunea exactă a serviciului Supabase și setările manuale din dashboard
  (activarea `vector`, alegerea pooler-ului). `[DE CONFIRMAT]`
- Comportamentul observat la suspendarea planurilor Free (Supabase/AuraDB). `[DE CONFIRMAT]`
- Versiunea Python pe Streamlit Community Cloud (lipsă `runtime.txt`). `[DE CONFIRMAT]`
- Numele aplicației și URL-ul public din Streamlit Community Cloud. `[DE CONFIRMAT]`
</content>
