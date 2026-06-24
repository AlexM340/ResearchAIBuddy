-- Migration 004: file metadata on documents (size + type) so Postgres can be
-- the single source of truth for the Notebooks UI (title, size, type, indexed).
-- Idempotent: safe to run multiple times.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_size BIGINT NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_type TEXT NOT NULL DEFAULT '';
