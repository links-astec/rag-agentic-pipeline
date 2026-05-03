-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table: stores chunks + embeddings + metadata
CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT        NOT NULL,
    embedding   VECTOR(384),           -- matches all-MiniLM-L6-v2 output dim
    source      TEXT,
    doc_type    TEXT,
    chunk_index INTEGER,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- IVFFlat index for approximate nearest-neighbour search
-- Lists = sqrt(expected row count). Tune after bulk insert.
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
