# scripts/eval — suită de evaluare (Capitolul Testare)

Automatizează partea **mecanică** a protocolului (rulare + capturare + calcul).
Partea de **judecată** (relevanța fragmentelor, calitatea răspunsului, utilitatea
grafului, sinteza, memoria) rămâne manuală.

Toate scripturile se rulează **din rădăcina proiectului** și folosesc același
config/DSN/Neo4j ca aplicația. Notebook-ul țintă: `Ai engineering Evaluation`
(suprascrie cu `EVAL_NOTEBOOK="..."`).

## Fluxul în 3 pași

### 1. Ground truth (manual asistat) — `label.py`
```bash
python scripts/eval/label.py            # construiește bazinul de candidați (top-30)
```
Generează `ground_truth.json` cu ~30 de chunk-uri candidate per întrebare F/X/C/G.
**Tu deschizi fișierul și pui `relevant: 1`** la chunk-urile care răspund efectiv
(citești câmpul `text`). Verifici acoperirea:
```bash
python scripts/eval/label.py --stats
```
Aceasta răspunde la „de unde știu care chunks sunt corecte”: nu adnotezi tot
corpusul, doar bazinul — tehnica standard *pooling* din IR.

### 2. Rulare — `run.py`
```bash
python scripts/eval/run.py              # 26 întrebări × 4 config (A-D)
```
Rulează cu `test_mode` ON (fără memorie personală), scope `topic` pe notebook,
golește cache-ul între configurații. Scrie `results/runs.jsonl` cu: rută, latență,
`retrieved_chunk_ids`, citări valide/invalide, `used_hyde/reranker/web`, graph_hits,
răspuns.

### 3. Raport — `report.py`
```bash
python scripts/eval/report.py
```
Calculează Recall@5, MRR, CitationValidity, latență (medie/P50/P95), route accuracy.
Scrie `results/report.md` (tabelele din capitol) + `results/details.csv` (anexă).

## Ce rămâne manual (nu se automatizează)
- marcarea relevanței în `ground_truth.json` (pasul 1);
- answer coverage (punctele necesare în răspuns);
- scorul grafului (−1/0/1/2) pe 5–10 întrebări C/G;
- memoria (M01–M05: 1/0.5/0), web-ul (N01–N04), sinteza/taxonomia (S, 1–5);
- alegerea celor 2–3 exemple discutate în text.

## Note
- Necesită `psycopg_pool` instalat (deja în `.venv`).
- Întrebările F/X/C/G/N/M sunt în `questions.json` (extrase din protocol).
- `results/` și `ground_truth.json` sunt date locale (nu le comite dacă nu vrei).
