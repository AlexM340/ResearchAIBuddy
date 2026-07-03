# Manual de utilizare — context (CerebrumAI)

Descriere factuală a fluxului real, derivată din `src/views/*` și `src/main_flash.py`.
Navigarea are 7 pagini, în bara de sus (`st.navigation`, `main_flash.py:835-855`):
**Second Brain**, **Notebooks**, **Notes**, **Tasks**, **Timeline**, **Knowledge Graph**, **Logs**.

## 0. Pornirea aplicației și onboarding

- La prima rulare fără `GOOGLE_API_KEY`, apare ecranul „Bun venit în CerebrumAI”
  care cere cheia Gemini (`main_flash.py:404-429`). Local, cheia se salvează în `.env`;
  pe Streamlit Cloud trebuie pusă în Secrets.
- După setarea cheii, sistemul se inițializează o singură dată și citește statusul
  indexului din Postgres (`bootstrap_documents_from_storage`, `main_flash.py:275-311`).

## 1. Second Brain (pagina principală)

Scop: chat global peste **tot** corpusul + memoria personală (`second_brain.py`).

- **Utilizator nou fără surse**: ghidaj în 3 pași (adaugă sursă → întreabă → vezi memoria),
  `_render_guided_start` (`second_brain.py:102-130`).
- **Conversație**: caseta de chat „Întreabă Second Brain...”. Întrebări-exemplu la
  început (`second_brain.py:38-42`).
- **Forțează web**: checkbox „Caută pe web pentru următoarea întrebare (Tavily)” —
  apare doar dacă Tavily e configurat (`second_brain.py:79-85`).
- **Unelte** (în sidebar, dialoguri modale — `second_brain.py:243-266`):
  - 📥 **Inbox**: propuneri de memorie (note/task/decizii/preferințe) detectate din
    conversații; le accepți sau le respingi (`_render_memory_inbox`).
  - 📊 **Insights**: statistici rapide, digest pe N zile, memorie recentă, decizii de revizuit.
  - ✏️ **Editor memorie**: CRUD pe decizii și preferințe.
  - ⚠️ **Contradicții**: relații `CONTRADICTS` din graf; dismiss/restore (necesită Neo4j).
  - 🔭 **Analize globale**: sinteză / taxonomie / gap analysis pe tot corpusul.
  - 🔎 **Sugestii surse**: surse propuse (necesită Tavily).
  - 🧹 **Curăță graful**: șterge muchii slabe, unește concepte, reconstruiește comunități (necesită Neo4j).
- **Remindere pinned**: banner cu task-uri expirate / scadente azi (`_render_pinned_reminders`).

## 2. Notebooks (spații de lucru pe topic)

Scop: surse + chat focusate pe un subiect (`notebooks.py`).

- **Creare/ștergere notebook**: formular „Nume notebook nou”; numele `general` e rezervat
  (`notebooks.py:100-123`). Notebook-ul se creează în Postgres (sursă de adevăr) + local staging
  (`_collections.py:76-88`).
- **Adăugare surse**: `st.file_uploader` cu **TXT / MD / PDF** (`notebooks.py:472-475`),
  butonul „Procesează” → salvare + indexare (chunking + embeddings), sincron, cu spinner
  (`process_uploaded_files`, `_documents.py:356-460`).
- **Lista de surse**: sortare (nume/dată/mărime/tip), filtre pe tip și „doar indexate”,
  ștergere per document (`notebooks.py:404-468`).
- **Conversație notebook**: chat propriu; checkbox „Doar surse din acest notebook”
  (mod `topic` vs `topic_general`) — `notebooks.py:501-507`. Opțional „Caută pe web”.
- **Sinteză topic**: agregare documente + note + decizii + preferințe într-un outline;
  se poate salva ca document indexat (`notebooks.py:202-303`).
- **Taxonomie + gap analysis** per notebook (`notebooks.py:306-391`).
- **Graf de cunoștințe (notebook)**: mini-graf scoped pe notebook (`render_mini_graph`).

### Surse web: persistente vs. temporare
- **Persistentă**: „salvarea unui link” descarcă pagina, curăță HTML, o salvează ca
  `web_*.txt` și o **indexează ca document** permanent (apare ca `[D#]`) —
  `add_web_url_as_document` (`_documents.py:119-192`).
- **Temporară**: bifa „Caută pe web (Tavily)” trimite întrebarea curentă către Tavily;
  rezultatele sunt folosite **doar pentru răspunsul acela** (`answer_origin = external`),
  nu se salvează ca sursă (`rag_module_flash.py:3258-3274`).

## 3. Formularea întrebărilor și citările

- Răspunsul vine cu mai multe tipuri de surse, afișate sub mesaj (`_shared.py:399-458`):
  - `[D#]` documente (`sources`),
  - `[G#]` surse din graf (concept → fișier),
  - `[M#]` memorie (artefacte: decizii/preferințe/episoade),
  - `[N#]` note proprii,
  - surse web externe (`external_sources`).
- `answer_origin` poate fi `internal` sau `external` (web) — afișat la final (`_shared.py:393,493`).
- Dacă nu există context suficient, modelul răspunde `INSUFFICIENT_CONTEXT`
  (`rag_module_flash.py:1766,3247-3250`).

## 4. Notes (idei proprii)

Scop: note scrise direct de utilizator, intrate în retrieval ca `[N#]` (`notes.py`).
Necesită Postgres (`notes.py:39-42`). Captură rapidă disponibilă și din sidebar
(„💡 Captureaza o idee”, `main_flash.py:541-582`).

## 5. Tasks

Scop: quick capture + kanban (Open / In progress / Done) + remindere (`tasks.py`).
Necesită Postgres (`tasks.py:38-41`). Task-urile pot fi propuse automat din conversații
(Inbox) sau adăugate manual.

## 6. Timeline

Scop: cronologie a evenimentelor de cunoștințe (note, decizii, task-uri, documente,
chat-uri) pe ferestre de 1–90 zile (`timeline.py`). Necesită Postgres.

## 7. Knowledge Graph

Scop: vizualizare interactivă Plotly a conceptelor și relațiilor din Neo4j (`graph.py`).
- Metrice: concepte, relații, contradicții, documente în graf.
- Filtre: tip relație, „doar contradicții”, notebook, număr maxim de noduri.
- Dacă Neo4j nu e disponibil, afișează un placeholder cu instrucțiuni (`graph.py:310-320`).

## 8. Logs (retrieval)

Scop: observabilitate — latențe (avg/p50/p95), distribuție pe rute (vector/memory/
hybrid/external/synthesis), tabel cu ultimele query-uri (`retrieval_logs.py`).
Necesită Postgres (`retrieval_logs.py:30-33`).

## 9. Memoria (cum „învață” asistentul)

- Din conversații se extrag automat decizii / preferințe / note / task-uri, propuse în
  **Inbox** (`memory_proposals`).
- Confirmarea/ștergerea se face în Inbox și în Editorul de memorie.
- „Mod testare (fără memorie personală)” (sidebar, `main_flash.py:653-662`) scoate
  M#/N#/Ep# din context și oprește capturarea — util pentru evaluări de retrieval.

## 10. Întreținere (sidebar → 🔧 Zona tehnică)

- „Aplică migrațiile DB”, „Backfill memory artifacts”, „Reconstruiește Neo4j din
  Postgres”, „Reindexează toate documentele”, „Migrează chat-urile locale în Postgres”
  (`main_flash.py:708-782`).
</content>
