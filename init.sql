-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS speakers (
    id SERIAL PRIMARY KEY,
    embedding vector(512),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create an index for vector similarity search
CREATE INDEX IF NOT EXISTS speakers_embedding_idx ON speakers USING ivfflat (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS audio_files (
    id SERIAL PRIMARY KEY,
    original_file_path TEXT NOT NULL UNIQUE,
    audio_duration_sec FLOAT NOT NULL,
    processing_window_start FLOAT NOT NULL,
    processing_window_end FLOAT NOT NULL,
    success BOOLEAN NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audio_chunks (
    id SERIAL PRIMARY KEY,
    audio_file_id INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
    chunk_file_path TEXT NOT NULL,
    start_time_ms INTEGER NOT NULL,  -- milliseconds
    end_time_ms INTEGER NOT NULL,    -- milliseconds
    transcription TEXT,
    transcription_conf FLOAT,
    transcription_clean TEXT,
    transcription_clean_conf FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audio_events (
    id SERIAL PRIMARY KEY,
    audio_file_id INTEGER NOT NULL REFERENCES audio_files(id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL REFERENCES audio_chunks(id) ON DELETE CASCADE,
    speaker_id INTEGER REFERENCES speakers(id),
    start_time_sec FLOAT NOT NULL,
    end_time_sec FLOAT NOT NULL,
    speaker_label TEXT NOT NULL,  -- The original speaker label from pyannote
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS audio_chunks_audio_file_id_idx ON audio_chunks (audio_file_id);
CREATE INDEX IF NOT EXISTS audio_events_audio_file_id_idx ON audio_events (audio_file_id);
CREATE INDEX IF NOT EXISTS audio_events_chunk_id_idx ON audio_events (chunk_id);
CREATE INDEX IF NOT EXISTS audio_events_speaker_id_idx ON audio_events (speaker_id);

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE parkiet TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres; 