# Publicare prin Streamlit Community Cloud (CerebrumAI)

## 1. Pe scurt

| Element | Valoare | Sursă |
|---|---|---|
| Fișier principal (entry point) | `src/main_flash.py` | `.devcontainer/devcontainer.json:22` |
| Dependențe | `requirements.txt` (rădăcină) | `requirements.txt` |
| Versiune Python | `[DE CONFIRMAT DE AUTOR]` — nu există `runtime.txt`/`.python-version` | — |
| Mecanism secrete | `os.getenv` (NU `st.secrets`) | grep global, `main_flash.py:176-181` |
| Repository | public, pe GitHub | cerință Streamlit Community Cloud |

## 2. Pașii de publicare

1. **Repository public pe GitHub** care conține `src/main_flash.py`,
   `requirements.txt`, `config.json` și folderul `src/`.
2. În Streamlit Community Cloud: New app → selectează repo + branch (`main`).
3. **Main file path**: `src/main_flash.py`.
4. **Secrets**: introdu-le în Settings → Secrets ale aplicației, ca **chei top-level**
   (vezi `.streamlit/secrets.toml.example`). Streamlit le expune ca variabile de
   mediu, iar codul le citește cu `os.getenv`:
   - `GOOGLE_API_KEY`
   - `POSTGRES_DSN`
   - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
   - `TAVILY_API_KEY`
5. **Deploy** și urmărește log-urile.
6. **Verifică conexiunile**: în sidebar → „Detalii sistem” trebuie să apară
   Postgres/Neo4j/Tavily ca „conectat” (`main_flash.py:501-513`).

> Notă tehnică: pentru ca `os.getenv` să vadă cheile, acestea trebuie să fie
> **top-level** în secrets TOML (nu în secțiuni `[...]`). Streamlit promovează doar
> cheile top-level în `os.environ`.

## 3. Dependențe

- `requirements.txt` este folosit automat de Streamlit Community Cloud.
- Conține `torch` + `sentence-transformers` (modelele de embeddings/reranker se
  descarcă la prima rulare) — pot crește timpul de pornire și consumul de memorie,
  relevant pe planul gratuit.

## 4. Limitările sistemului de fișiere

Aplicația scrie pe disc local (toate sub `./data/...`):
- bibliotecă documente: `./data/document_library/` (`document_manager.py:19-32`);
- fișiere uploadate: `./data/uploaded_docs/` (`_documents.py:380`);
- chat-uri locale JSON: `./data/chat_sessions/` (`chat_session_manager.py:26`);
- cache-uri: `./data/cache/`, `./data/embeddings_cache/`.

Pe Streamlit Community Cloud filesystem-ul **nu este persistent** între reporniri/
redeploy-uri. Implicații, derivate din cod:

- **Conținutul indexat persistă** dacă `POSTGRES_DSN` e configurat: chunks +
  embeddings + metadatele documentelor stau în Supabase, iar UI-ul citește din
  Postgres ca sursă de adevăr (`_collections.py:1-73`). Deci căutarea/răspunsurile
  funcționează și după repornire.
- **Fișierele brute uploadate NU persistă**: copia din `./data/uploaded_docs` și
  `./data/document_library` se pierde. Funcția „Reindexează toate documentele”
  (`_documents.py:71-116`) are nevoie de fișierele locale și **poate eșua** după o
  repornire pentru documentele ale căror fișiere brute au dispărut.
- **Chat-urile locale (JSON) NU persistă** decât dacă sunt migrate în Postgres
  („Migrează în Postgres”, `main_flash.py:319-390`).
- **`.env` nu există pe Cloud**: secretele vin din dashboard ca variabile de mediu.
  Ecranul de onboarding care scrie cheia în `.env` (`main_flash.py:416-424`) este util
  doar local — pe Cloud cheia trebuie pusă în Secrets.

## 5. Verificarea deployment-ului

1. Aplicația se încarcă fără ecranul de onboarding (cheia Gemini e setată).
2. Sidebar → pastila „🟢 Conectat” = Postgres OK (`main_flash.py:489`).
3. Sidebar → „Detalii sistem”: Postgres/Neo4j/Tavily „conectat”.
4. Tab Notebooks: creează un notebook, încarcă un PDF/TXT/MD, „Procesează” →
   sursa apare ca „indexat”.
5. Tab Second Brain: pune o întrebare → primești răspuns cu citări.

## 6. Probleme posibile la deployment

- **Build lent / memorie**: `torch` + modele HF descărcate la runtime.
- **Ingestie sincronă**: upload-uri mari blochează UI-ul (fără background).
- **Conexiuni Supabase Free**: folosește pooler-ul (port 6543, mod tranzacție);
  codul e pregătit (`prepare_threshold=None`, `postgres_client.py:76-80`).
- **Supabase/AuraDB Free în pauză**: instanțele gratuite se suspendă după
  inactivitate → conexiunea eșuează până la reactivare (vezi `errors_and_limitations.md`).
- **Versiune Python pe Cloud**: fără `runtime.txt`, versiunea e cea implicită a
  platformei → `[DE CONFIRMAT DE AUTOR]`.
</content>
