# Variabile de mediu (CerebrumAI)

Aplicația citește variabilele **exclusiv prin `os.getenv`** (nu folosește
`st.secrets`). Local, `.env` din rădăcină este încărcat cu `load_dotenv`
(`src/main_flash.py:35-37`). Pe Streamlit Community Cloud, secretele definite în
dashboard (la nivel top, în TOML) sunt expuse ca variabile de mediu și sunt citite
de același `os.getenv`.

Pentru fiecare variabilă, dacă lipsește din env, se folosește valoarea
corespunzătoare din `config.json` (vezi coloana „Fallback config”).

## Tabel variabile

| Variabilă | Oblig./Opț. | Serviciu | Citită în | Fallback config | Exemplu sigur | Comportament dacă lipsește |
|---|---|---|---|---|---|---|
| `GOOGLE_API_KEY` | **Obligatorie** | Gemini (LLM) | `main_flash.py:400,873` | `config.json → gemini_api_key` (nefolosit direct la bootstrap) | `AIzaSy...` | Ecran de onboarding; `st.stop()` — aplicația nu pornește (`main_flash.py:867-868`) |
| `POSTGRES_DSN` | Opțională (necesară pt. Second Brain complet) | Supabase PostgreSQL | `main_flash.py:178` | `config.json → storage.postgres_dsn` | `postgresql://user:password@host:6543/postgres?sslmode=require` | Mod local fallback; Notes/Tasks/Timeline/Logs și memoria persistentă indisponibile (`main_flash.py:495-496`) |
| `NEO4J_URI` | Opțională | Neo4j AuraDB | `main_flash.py:179` | `config.json → graph.neo4j_uri` | `neo4j+s://xxxxxxxx.databases.neo4j.io` | Graful dezactivat (vector-only) |
| `NEO4J_USER` | Opțională | Neo4j AuraDB | `main_flash.py:180` | `config.json → graph.neo4j_user` | `neo4j` | Graful dezactivat |
| `NEO4J_PASSWORD` | Opțională | Neo4j AuraDB | `main_flash.py:181` | `config.json → graph.neo4j_password` | `example_password` | Graful dezactivat |
| `TAVILY_API_KEY` | Opțională | Tavily (web search) | `main_flash.py:176`, `rag_module_flash.py:3255,3264` | `config.json → tavily_api_key` | `tvly-...` | Web search dezactivat (checkbox „Caută pe web” ascuns) |
| `EVAL_NOTEBOOK` | Opțională (doar scripturi eval) | — | `scripts/eval/_bootstrap.py:35` | `"Ai engineering Evaluation"` | `numele unui notebook` | Folosește valoarea implicită; irelevant pentru aplicația web |

> Important: codul citește `NEO4J_USER` (NU `NEO4J_USERNAME`), `POSTGRES_DSN`
> (NU `DATABASE_URL`) și `GOOGLE_API_KEY` (NU `GEMINI_API_KEY`). Acestea sunt
> numele reale, confirmate în `main_flash.py:176-181,400`.

## Condiții de activare a serviciilor

- **Postgres** este `enabled` doar dacă: DSN ne-gol **și** `psycopg` instalat
  (`postgres_client.py:45`).
- **Neo4j** este `enabled` doar dacă: URI + user + parolă ne-goale **și** driverul
  `neo4j` instalat **și** inițializarea driverului reușește (`neo4j_client.py:27,33-39`).
- **Tavily/web search** este disponibil doar dacă există o cheie ne-goală
  (`rag_module_flash.py:3252-3256`).

## Precedența valorilor

Pentru cheile cu fallback (Tavily, Postgres, Neo4j), ordinea este:
`os.getenv(VAR)` → `config.json` → string gol (`main_flash.py:176-181`).
Pentru `GOOGLE_API_KEY`, doar env (sau salvată în `.env` prin ecranul de onboarding,
`main_flash.py:416-424`).

## Securitate

- Nu comite niciodată valori reale în `.env` sau `config.json`.
- `.env`, `.env.local`, `.env.production` sunt deja în `.gitignore` (`.gitignore:39-41`).
- `config.json` **este** urmărit în git și are câmpuri pentru secrete goale, dar
  conține un `graph.neo4j_uri` specific proiectului (`config.json:46`) — vezi
  `open_questions.md`. Recomandare: golește-l / mută-l în env înainte de publicare.
</content>
