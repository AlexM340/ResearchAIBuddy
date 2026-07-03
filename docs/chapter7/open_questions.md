# Întrebări deschise pentru autor (CerebrumAI)

Informații care NU pot fi stabilite din cod și trebuie completate manual înainte de
redactarea Capitolului 7.

## Infrastructură și deployment
1. **Versiunea Python pe Streamlit Community Cloud** — repo-ul nu conține `runtime.txt`
   sau `.python-version`. Devcontainer-ul folosește 3.11. Ce versiune rulează efectiv pe Cloud?
2. **Numele aplicației și URL-ul public** din Streamlit Community Cloud (de inclus, fără
   date de cont).
3. **Branch-ul** folosit pentru deployment (probabil `main`) — de confirmat.

## Supabase
4. **Pașii manuali din dashboard**: a fost activată extensia `vector` manual? S-a folosit
   connection pooler-ul (port 6543, mod tranzacție) sau conexiunea directă (5432)?
5. **Forma exactă a DSN-ului** (include `?sslmode=require`?) — fără a dezvălui valori reale.
6. **Comportament observat** la suspendarea instanței Free după inactivitate.

## Neo4j AuraDB
7. **Confirmare** că URI-ul este de forma `neo4j+s://...` și utilizatorul `neo4j`.
8. **Comportament observat** la auto-pauză/ștergere a instanței Free.

## Cod vs. configurație (contradicții / curățenie de verificat)
9. `config.json:46` conține un `graph.neo4j_uri` **specific proiectului**
   (`neo4j+s://...databases.neo4j.io`). Trebuie golit / mutat în variabilă de mediu
   înainte de publicarea repo-ului? (recomandat: da)
10. `config.json` are câmpuri prezente dar aparent **neutilizate** de fluxul principal:
    `vector_db.type: "faiss"`, `openai_api_key`, `wandb_api_key`, `models.local_llm`,
    `agent.*`, `features.web_search: false`, `features.multimodal`. De clarificat care
    sunt active vs. moștenite (legacy), ca să nu fie descrise greșit în capitol.
11. `requirements.txt` listează `python-docx`, `python-pptx`, `openpyxl`, `unstructured`,
    `chromadb`, `faiss-cpu`, dar procesorul de documente acceptă efectiv doar **TXT/MD/PDF**
    (`rag_module_flash.py:869-884`) și UI-ul doar TXT/MD/PDF. De confirmat că DOCX/XLSX/PPTX
    **nu** sunt suportate în versiunea finală (sau dacă există o cale care le folosește).

## Resurse vizuale (pentru capitol)
12. **Capturi de ecran** necesare: ecran onboarding, Second Brain (chat + surse),
    Notebooks (upload + procesare), Knowledge Graph, Logs, sidebar status „Detalii sistem”.
13. Eventual o diagramă a fluxului de date (există deja `docs/architecture_design.md`).
</content>
