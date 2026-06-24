# scripts/

Utilitare de mentenanță a datelor pentru CerebrumAI — rulate **manual**, one-shot,
idempotente. Nu fac parte din fluxul aplicației și nu sunt importate de cod.

> Notă: acestea NU sunt migrările de schemă. Migrările de schemă stau în
> `src/storage/migrations/*.sql` și sunt aplicate **automat** la pornire de
> `postgres_client.run_migrations()` (urmărite în tabela `schema_migrations`).
> Nu le muta de acolo.

## Cum se rulează

Din **rădăcina proiectului** (scripturile urcă singure un nivel ca să găsească
`src/`, `config.json` și `.env`):

```bash
python scripts/<nume_script>.py --dry-run   # raporteaza, nu scrie nimic
python scripts/<nume_script>.py             # aplica
```

Conexiunea la Postgres se ia din `POSTGRES_DSN` (env / `.env`) sau, ca fallback,
din `config.json` → `storage.postgres_dsn`.

## Scripturi

### `migrate_library_to_postgres.py`
Migrează biblioteca locală (`./data/document_library/library_metadata.json`,
gestionată de `DocumentManager`) în Postgres: face **upsert** pe colecții și pe
rândurile din `documents`, astfel încât deploy-ul de pe Streamlit (care citește
din Postgres) să vadă aceleași notebook-uri.

- Idempotent (`ON CONFLICT ... DO UPDATE`), fără ștergeri.
- Migrează **doar metadate** (colecții + documente), **nu** chunks/embeddings.
  Un document local neindexat în Postgres va apărea ca sursă, dar fără conținut
  pentru retrieval până la o re-indexare din aplicație.

Opțiuni: `--library <cale>` (implicit `./data/document_library`), `--dry-run`.

### `reconcile_indexed_flags.py`
Reconciliază flagul `documents.indexed` cu realitatea (prezența chunk-urilor):

```
indexed = TRUE  daca documentul are >= 1 chunk
indexed = FALSE daca nu are niciun chunk
```

Atinge doar coloana boolean `documents.indexed` — nu modifică chunks/embeddings.
Util după o migrare sau ingestii în masă, ca să nu rămână documente marcate
`indexed=true` dar goale (sărite la reindexare) sau invers. La final listează
documentele care mai trebuie re-indexate (0 chunks).

Opțiuni: `--dry-run`.

## Ordine recomandată

1. `migrate_library_to_postgres.py` — aduce colecțiile/documentele în Postgres.
2. `reconcile_indexed_flags.py` — aliniază flagul `indexed` cu chunk-urile reale.
