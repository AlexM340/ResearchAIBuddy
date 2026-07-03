"""Sterge episoadele auto-capturate din Postgres (curatare pentru testare).

Episoadele sunt memorie conversationala generata automat la fiecare Q&A
(`memory_artifacts` cu artifact_type='episode'). Pentru un set de testare curat
le poti elimina fizic. Embeddings-urile asociate se sterg prin ON DELETE CASCADE.

NU atinge documentele, chunk-urile, notele scrise de tine sau istoricul de chat
(`messages`). Pentru chat-uri, foloseste butonul "Sterge" din UI.

Utilizare (din radacina proiectului):
    python scripts/clear_episodes.py --dry-run            # cate ar sterge (implicit recomandat)
    python scripts/clear_episodes.py                      # sterge TOATE episoadele
    python scripts/clear_episodes.py --topic "Ai engineering Evaluation"   # doar un topic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from storage import PostgresClient  # noqa: E402


def _dsn() -> str:
    dsn = os.getenv("POSTGRES_DSN", "").strip()
    if not dsn:
        cfg = ROOT / "config.json"
        if cfg.exists():
            try:
                dsn = (json.loads(cfg.read_text(encoding="utf-8")).get("storage", {}).get("postgres_dsn") or "").strip()
            except Exception:
                pass
    return dsn


def main() -> int:
    parser = argparse.ArgumentParser(description="Sterge episoadele auto-capturate")
    parser.add_argument("--dry-run", action="store_true", help="Doar raporteaza; nu sterge")
    parser.add_argument("--topic", default="", help="Limiteaza la un topic_collection (implicit: toate)")
    args = parser.parse_args()

    dsn = _dsn()
    if not dsn:
        print("EROARE: POSTGRES_DSN lipseste (env sau config.json -> storage.postgres_dsn).")
        return 2

    client = PostgresClient(dsn=dsn, embedding_dim=384)
    if not client.enabled or not client.test_connection():
        print("EROARE: nu ma pot conecta la Postgres.")
        return 2

    topic = (args.topic or "").strip()
    where = "artifact_type = 'episode'"
    params: tuple = ()
    if topic:
        where += " AND topic_collection = %s"
        params = (topic,)

    with client.connection() as conn:
        cur = conn.cursor()
        # Numarare pe topic, pentru transparenta.
        cur.execute(
            f"SELECT COALESCE(topic_collection, ''), COUNT(*) FROM memory_artifacts WHERE {where} GROUP BY topic_collection ORDER BY 2 DESC;",
            params,
        )
        rows = cur.fetchall() or []
        total = sum(int(r[1]) for r in rows)

        print(f"Episoade gasite{f' in topic {topic!r}' if topic else ''}: {total}")
        for name_col, count in rows:
            print(f"  {name_col or '(fara topic)':<32} {count}")

        if total == 0:
            print("Nimic de sters.")
            return 0

        if args.dry_run:
            print("\n[DRY-RUN] Nu s-a sters nimic. Reruleaza fara --dry-run ca sa aplici.")
            return 0

        # Sterge evenimentele de tip episode (cosmetic) + artefactele (cascade embeddings).
        cur.execute(
            "DELETE FROM memory_events WHERE event_type = 'episode_captured'"
            + (" AND topic_collection = %s" if topic else "") + ";",
            params if topic else (),
        )
        cur.execute(f"DELETE FROM memory_artifacts WHERE {where};", params)
        deleted = cur.rowcount
        conn.commit()
        print(f"\nSters: {deleted} episoade (+ embeddings prin CASCADE).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
