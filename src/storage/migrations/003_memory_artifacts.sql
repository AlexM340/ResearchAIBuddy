-- Migration 003: canonical memory artifacts, events and proposals.
-- Idempotent: safe to run multiple times.

CREATE TABLE IF NOT EXISTS memory_artifacts (
    id BIGSERIAL PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    source_table TEXT NOT NULL DEFAULT '',
    source_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    topic_collection TEXT NOT NULL DEFAULT '',
    tags TEXT[] NOT NULL DEFAULT '{}',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'active',
    source_message_id BIGINT NULL REFERENCES messages(id) ON DELETE SET NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (artifact_type, source_table, source_id)
);

CREATE TABLE IF NOT EXISTS memory_artifact_embeddings (
    artifact_id BIGINT PRIMARY KEY REFERENCES memory_artifacts(id) ON DELETE CASCADE,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memory_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    artifact_type TEXT NOT NULL DEFAULT '',
    artifact_id BIGINT NULL REFERENCES memory_artifacts(id) ON DELETE SET NULL,
    topic_collection TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    preview TEXT NOT NULL DEFAULT '',
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memory_proposals (
    id BIGSERIAL PRIMARY KEY,
    proposal_type TEXT NOT NULL,
    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'pending',
    source_message_id BIGINT NULL REFERENCES messages(id) ON DELETE SET NULL,
    topic_collection TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_artifacts_type_topic_updated
ON memory_artifacts(artifact_type, topic_collection, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_artifacts_source
ON memory_artifacts(source_table, source_id);
CREATE INDEX IF NOT EXISTS idx_memory_artifacts_fts
ON memory_artifacts USING GIN (to_tsvector('simple', coalesce(title, '') || ' ' || content));
CREATE INDEX IF NOT EXISTS idx_memory_artifact_embeddings_hnsw
ON memory_artifact_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memory_events_topic_time
ON memory_events(topic_collection, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_events_type_time
ON memory_events(event_type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_proposals_status_time
ON memory_proposals(status, created_at DESC);
