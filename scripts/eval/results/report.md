# Raport evaluare CerebrumAI (A-D)

## Tabel principal (medii pe cele 26 de intrebari F/X/C/G)

| Configurație | Recall@5 | MRR | Citări valide | Latență medie (ms) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| A – vectorial | 0.171 | 0.253 | 1.000 | 10052 | 6965 | 22988 |
| B – reranker | 0.301 | 0.447 | 1.000 | 9809 | 8165 | 17356 |
| C – HyDE + reranker | 0.342 | 0.464 | 1.000 | 11065 | 11080 | 14729 |
| D – sistem complet | 0.285 | 0.429 | 1.000 | 11480 | 11479 | 15359 |

## Recall@5 / MRR pe categorii

| Configurație | F (Recall@5 / MRR) | X (Recall@5 / MRR) | C (Recall@5 / MRR) | G (Recall@5 / MRR) |
|---|---:|---:|---:|---:|
| A – vectorial | 0.13 / 0.20 | 0.24 / 0.32 | 0.11 / 0.25 | 0.25 / 0.30 |
| B – reranker | 0.38 / 0.48 | 0.41 / 0.65 | 0.12 / 0.33 | 0.25 / 0.31 |
| C – HyDE + reranker | 0.50 / 0.54 | 0.24 / 0.37 | 0.20 / 0.42 | 0.30 / 0.47 |
| D – sistem complet | 0.42 / 0.45 | 0.23 / 0.40 | 0.28 / 0.58 | 0.07 / 0.24 |

## Route accuracy (informativ; relevant mai ales pentru D)

| Configurație | Route accuracy |
|---|---:|
| A – vectorial | 0.462 |
| B – reranker | 0.385 |
| C – HyDE + reranker | 0.462 |
| D – sistem complet | 0.462 |

