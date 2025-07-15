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
    audio_file_id INTEGER NOT NULL REFERENCES audio_files(id),
    chunk_file_path TEXT NOT NULL,
    start_time_ms INTEGER NOT NULL,  -- milliseconds
    end_time_ms INTEGER NOT NULL,    -- milliseconds
    transcription TEXT,
    transcription_clean TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audio_events (
    id SERIAL PRIMARY KEY,
    audio_file_id INTEGER NOT NULL REFERENCES audio_files(id),
    speaker_id INTEGER REFERENCES speakers(id),
    start_time_sec FLOAT NOT NULL,
    end_time_sec FLOAT NOT NULL,
    speaker_label TEXT NOT NULL,  -- The original speaker label from pyannote
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunk_event_links (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES audio_chunks(id),
    event_id INTEGER NOT NULL REFERENCES audio_events(id),
    event_start_in_chunk_sec FLOAT NOT NULL,  -- When the event starts relative to chunk start
    event_end_in_chunk_sec FLOAT NOT NULL,   -- When the event ends relative to chunk start
    event_duration_in_chunk_sec FLOAT NOT NULL, -- Duration of this event within the chunk
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS audio_chunks_audio_file_id_idx ON audio_chunks (audio_file_id);
CREATE INDEX IF NOT EXISTS audio_events_audio_file_id_idx ON audio_events (audio_file_id);
CREATE INDEX IF NOT EXISTS audio_events_speaker_id_idx ON audio_events (speaker_id);
CREATE INDEX IF NOT EXISTS chunk_event_links_chunk_id_idx ON chunk_event_links (chunk_id);
CREATE INDEX IF NOT EXISTS chunk_event_links_event_id_idx ON chunk_event_links (event_id);

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE parkiet TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres; 