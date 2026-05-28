CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS collections (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL DEFAULT 'topic',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    file_hash TEXT NOT NULL UNIQUE,
    original_name TEXT NOT NULL,
    collection_id BIGINT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    source_path TEXT NOT NULL,
    indexed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_order INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chats (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    topic_collection TEXT NOT NULL DEFAULT '',
    query_mode TEXT NOT NULL DEFAULT 'topic_general',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    answer_origin TEXT NOT NULL DEFAULT 'internal',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS decisions (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    rationale TEXT NOT NULL,
    topic_collection TEXT NOT NULL DEFAULT '',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    source_message_id BIGINT NULL REFERENCES messages(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS preferences (
    id BIGSERIAL PRIMARY KEY,
    preference_key TEXT NOT NULL,
    preference_value TEXT NOT NULL,
    topic_collection TEXT NOT NULL DEFAULT '',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    source_message_id BIGINT NULL REFERENCES messages(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (preference_key, topic_collection)
);

CREATE TABLE IF NOT EXISTS tasks (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT '',
    topic_collection TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    priority TEXT NOT NULL DEFAULT 'normal',
    due_at TIMESTAMPTZ NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    source_message_id BIGINT NULL REFERENCES messages(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    route_used TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- User-authored notes: the "Second Brain" primitive for capturing your own thinking
-- (not extracted from documents or chat — written directly by the user).
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

CREATE INDEX IF NOT EXISTS idx_chunks_document_order ON chunks(document_id, chunk_order);
CREATE INDEX IF NOT EXISTS idx_messages_chat_created ON messages(chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_decisions_topic_created ON decisions(topic_collection, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_preferences_topic_updated ON preferences(topic_collection, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_topic_status_updated ON tasks(topic_collection, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_notes_topic_updated ON notes(topic_collection, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notes_fts ON notes USING GIN (to_tsvector('simple', coalesce(title, '') || ' ' || content));
CREATE INDEX IF NOT EXISTS idx_chunks_fts ON chunks USING GIN (to_tsvector('simple', text));
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw
ON chunk_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_note_embeddings_hnsw
ON note_embeddings USING hnsw (embedding vector_cosine_ops);
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
