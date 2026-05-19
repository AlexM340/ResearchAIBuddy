-- Migration 002: Notes primitive — user-authored Second Brain content.
-- Idempotent: safe to run multiple times.

CREATE TABLE IF NOT EXISTS notes (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    topic_collection TEXT NOT NULL DEFAULT '',
    tags TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS note_embeddings (
    note_id BIGINT PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_notes_topic_updated ON notes(topic_collection, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notes_fts ON notes USING GIN (to_tsvector('simple', coalesce(title, '') || ' ' || content));
CREATE INDEX IF NOT EXISTS idx_note_embeddings_hnsw
ON note_embeddings USING hnsw (embedding vector_cosine_ops);
