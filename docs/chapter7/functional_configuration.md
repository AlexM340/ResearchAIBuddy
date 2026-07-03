# Configurația funcțională (CerebrumAI)

Setările funcționale stau în `config.json` (rădăcină), încărcate de `load_config()`
și mapate în obiectul de configurare prin `build_apci_config_dict()`
(`src/main_flash.py:97-231`). Valorile din `config.json` au prioritate peste
`_default_config()`; pentru chei sensibile (Tavily, Postgres, Neo4j) variabilele de
mediu au prioritate peste `config.json`.

| Setare | Fișier | Cheie / câmp | Valoare implicită (config.json) | Schimbabilă prin |
|---|---|---|---|---|
| Model LLM principal | `config.json` | `models.primary_llm` | `gemini-2.5-flash` | config |
| Model LLM fallback | `config.json` | `models.fallback_llm` | `gemini-2.0-flash-exp` | config |
| Model embeddings | `config.json` | `models.embedding_model` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | config |
| Dimensiune embeddings | `config.json` | `storage.pgvector_embedding_dim` | `384` | config (trebuie să corespundă modelului + `VECTOR(384)` din schema) |
| Dimensiune fragment (chunk) | `config.json` | `chunking.chunk_size` | `1000` | config |
| Suprapunere fragmente | `config.json` | `chunking.chunk_overlap` | `200` | config |
| Documente max în context | `config.json` | `chunking.max_context_docs` | `5` | config |
| Temperatură Gemini | `config.json` | `gemini.temperature` | `0.1` | config |
| Max tokens Gemini | `config.json` | `gemini.max_tokens` | `65536` | config |
| Rate limit (rpm / rpd) | `config.json` | `gemini.rate_limits.requests_per_minute` / `requests_per_day` | `300` / `10000` | config (default cod: 30 / 3000) |
| Candidați retrieval | `config.json` | `rag.retrieval_candidates` | `20` | config |
| Rerank top-k | `config.json` | `rag.rerank_top_k` | `8` | config |
| Hybrid alpha | `config.json` | `rag.hybrid_alpha` | `0.7` | config |
| HyDE (activare) | `config.json` | `retrieval_quality.enable_hyde` | `true` | config + **toggle UI** (sidebar) |
| HyDE max tokens | `config.json` | `retrieval_quality.hyde_max_tokens` | `200` | config |
| Reranker (activare) | `config.json` | `retrieval_quality.enable_reranker` | `true` | config + **toggle UI** |
| Model reranker | `config.json` | `retrieval_quality.reranker_model` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | config |
| Reranker input-k | `config.json` | `retrieval_quality.reranker_input_k` | `20` | config |
| Graph RAG (activare) | `config.json` | `second_brain.enable_graph_rag` | `true` | config + **toggle UI** |
| Extragere structurată graf | `config.json` | `second_brain.graph_extraction_mode` | `llm_structured` | config |
| Batch extragere graf | `config.json` | `second_brain.graph_extraction_batch_size` | `12` (default cod: 5) | config |
| Hops max în graf | `config.json` | `second_brain.graph_max_hops` | `2` | config |
| Încredere minimă graf | `config.json` | `second_brain.graph_min_confidence` | `0.7` | config |
| Prag încredere vector (router) | `config.json` | `second_brain.vector_confidence_threshold` | `0.35` | config |
| Buget context (tokens) | `config.json` | `second_brain.context_budget_tokens` | `8000` | config |
| Raport buget vector / graf | `config.json` | `second_brain.vector_budget_ratio` / `graph_budget_ratio` | `0.65` / `0.35` | config |
| Router mode | `config.json` | `second_brain.router_mode` | `rule_based` | config |
| Memorie: consolidare | `config.json` | `second_brain.memory_consolidation_enabled` | `true` | config |
| Memorie: decay (zile) | `config.json` | `second_brain.memory_decay_days` | `90` | config |
| DB primary strict | `config.json` | `second_brain.db_primary_strict` | `true` | config |
| Ingestie: batch embeddings | `config.json` | `second_brain.ingestion_embedding_batch_size` | `64` | config |
| Ingestie: workers | `config.json` | `second_brain.ingestion_max_workers` | `4` | config |
| Tavily (web fallback) | `config.json` | `web_search.enable_web_fallback` | `true` | config + **toggle UI** + env `TAVILY_API_KEY` |
| Tavily: rezultate max | `config.json` | `web_search.max_results` | `5` | config |
| Tavily: adâncime căutare | `config.json` | `web_search.search_depth` | `advanced` | config |
| Vector top-k | `config.json` | `retrieval.vector_top_k` | `8` | config |
| Graf top-k căi | `config.json` | `retrieval.graph_top_k_paths` | `6` | config |
| Cache răspunsuri | cod | `cache_enabled` (hardcodat True) | `true` | cod (`rag_module_flash.py:1421`) |

## Note importante

- **Extragerea structurată în graf** este implicit `llm_structured` (consumă apeluri
  LLM la ingestie). Confirmat în `config.json:72` și `main_flash.py:210`.
- **Toggle-uri din UI** (sidebar → ⚙️ Setări, `main_flash.py:619-662`): HyDE, Reranker,
  Graph RAG și „Mod testare”. Au efect imediat la următoarea întrebare; **nu** reconstruiesc
  ingestia/graful deja create. Configurațiile de test sunt etichetate A/B/C/D
  (`_test_config_label`, `main_flash.py:585-593`).
- **Praguri de încredere**: `graph_min_confidence` (0.7), `vector_confidence_threshold`
  (0.35) și pragul de curățare a grafului (slider 0.50–0.90, implicit 0.78,
  `second_brain.py:333-336`).
- **Reranker / model embeddings**: dacă modelul de embeddings se schimbă față de
  ultima indexare, apare un banner de reindexare (`main_flash.py:790-821`).
- `config.json` conține și câmpuri prezente dar **neutilizate** de fluxul principal
  (ex. `vector_db.type: faiss`, `agent.*`, `features.web_search: false`) — vezi
  `open_questions.md`.
</content>
