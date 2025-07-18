import logging
import numpy as np
from parkiet.audioprep.schemas import (
    ProcessedAudioFile,
    ProcessedAudioChunk,
    SpeakerEvent,
)
from parkiet.database.connection import get_db_connection

log = logging.getLogger(__name__)


class AudioStore:
    """Database operations for storing audio processing data."""

    def __init__(self, similarity_threshold: float = 0.5, db_connection=None):
        self.similarity_threshold = similarity_threshold
        self.db = db_connection or get_db_connection()

    def _cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)

    def _find_similar_speaker(self, embedding: np.ndarray) -> int | None:
        with self.db.get_cursor() as cursor:
            embedding_list = embedding.tolist()
            cursor.execute(
                """
                SELECT id, 1 - (embedding <=> %s::vector) as similarity
                FROM speakers
                WHERE 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT 1
            """,
                (
                    embedding_list,
                    embedding_list,
                    self.similarity_threshold,
                    embedding_list,
                ),
            )
            result = cursor.fetchone()
            if result:
                return result["id"]  # type: ignore
        return None

    def store_audio_file(self, processed_file: ProcessedAudioFile) -> int:
        """
        Store audio file metadata in the database.

        Args:
            processed_file: ProcessedAudioFile object

        Returns:
            audio_file_id: The ID of the inserted audio file record
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audio_files 
                (original_file_path, audio_duration_sec, processing_window_start, 
                 processing_window_end, success)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (original_file_path) DO UPDATE SET
                    audio_duration_sec = EXCLUDED.audio_duration_sec,
                    processing_window_start = EXCLUDED.processing_window_start,
                    processing_window_end = EXCLUDED.processing_window_end,
                    success = EXCLUDED.success
                RETURNING id
            """,
                (
                    processed_file.gcs_audio_path,
                    processed_file.audio_duration_sec,
                    processed_file.processing_window.get("start", 0.0),
                    processed_file.processing_window.get(
                        "end", processed_file.audio_duration_sec
                    ),
                    processed_file.success,
                ),
            )
            return cursor.fetchone()["id"]  # type: ignore

    def is_audio_file_processed(self, file_path: str) -> bool:
        """
        Check if an audio file has already been processed.

        Args:
            file_path: Path to the audio file

        Returns:
            True if the file has been processed, False otherwise
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, success FROM audio_files 
                WHERE original_file_path = %s
            """,
                (file_path,),
            )

            result = cursor.fetchone()
            if result:
                # File exists in database, check if it was processed successfully
                return result["success"]  # type: ignore
            return False

    def store_speaker_embeddings(
        self, speaker_embeddings: dict[str, np.ndarray]
    ) -> dict[str, int]:
        """
        Store speaker embeddings and return mapping of speaker labels to speaker IDs.
        Only inserts new embeddings if they are distinct enough from existing ones.

        Args:
            speaker_embeddings: Dictionary mapping speaker labels to embeddings

        Returns:
            speaker_id_map: Dictionary mapping speaker labels to speaker IDs
        """
        speaker_id_map = {}

        with self.db.get_cursor() as cursor:
            for speaker_label, embedding in speaker_embeddings.items():
                # Check if we have a similar speaker already
                similar_speaker_id = self._find_similar_speaker(embedding)

                if similar_speaker_id is not None:
                    # Use existing speaker ID
                    speaker_id_map[speaker_label] = similar_speaker_id
                    log.debug(
                        f"Reusing speaker ID {similar_speaker_id} for label {speaker_label}"
                    )
                else:
                    # Insert new speaker embedding
                    embedding_list = embedding.tolist()

                    cursor.execute(
                        """
                        INSERT INTO speakers (embedding)
                        VALUES (%s::vector)
                        RETURNING id
                    """,
                        (embedding_list,),
                    )

                    speaker_id = cursor.fetchone()["id"]  # type: ignore
                    speaker_id_map[speaker_label] = speaker_id
                    log.info(f"Stored new speaker {speaker_label} with ID {speaker_id}")

        return speaker_id_map

    def store_audio_events(
        self,
        audio_file_id: int,
        speaker_events: list[SpeakerEvent],
        speaker_id_map: dict[str, int] | None = None,
    ) -> list[int]:
        """
        Store audio events in the database.

        Args:
            audio_file_id: ID of the audio file
            speaker_events: List of speaker events
            speaker_id_map: Optional mapping of speaker labels to speaker IDs

        Returns:
            event_ids: List of event IDs that were inserted
        """
        event_ids = []

        with self.db.get_cursor() as cursor:
            for event in speaker_events:
                speaker_id = None
                if speaker_id_map and event.speaker in speaker_id_map:
                    speaker_id = speaker_id_map[event.speaker]

                cursor.execute(
                    """
                    INSERT INTO audio_events 
                    (audio_file_id, speaker_id, start_time_sec, end_time_sec, speaker_label)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (audio_file_id, speaker_id, event.start, event.end, event.speaker),
                )

                event_id = cursor.fetchone()["id"]  # type: ignore
                event_ids.append(event_id)

        log.info(f"Stored {len(event_ids)} audio events for audio file {audio_file_id}")
        return event_ids

    def store_audio_chunks(
        self, audio_file_id: int, processed_chunks: list[ProcessedAudioChunk]
    ) -> list[int]:
        """
        Store audio chunks in the database.

        Args:
            audio_file_id: ID of the audio file
            processed_chunks: List of processed audio chunks

        Returns:
            chunk_ids: List of chunk IDs that were inserted
        """
        chunk_ids = []

        with self.db.get_cursor() as cursor:
            for chunk in processed_chunks:
                cursor.execute(
                    """
                    INSERT INTO audio_chunks 
                    (audio_file_id, chunk_file_path, start_time_ms, end_time_ms, transcription, transcription_conf, transcription_clean, transcription_clean_conf)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (
                        audio_file_id,
                        chunk.audio_chunk.gcs_file_path or chunk.audio_chunk.file_path,
                        chunk.audio_chunk.start,
                        chunk.audio_chunk.end,
                        chunk.transcription,
                        chunk.transcription_conf,
                        chunk.transcription_clean,
                        chunk.transcription_clean_conf,
                    ),
                )

                chunk_id = cursor.fetchone()["id"]
                chunk_ids.append(chunk_id)

        log.info(f"Stored {len(chunk_ids)} audio chunks for audio file {audio_file_id}")
        return chunk_ids

    def store_chunk_event_links(
        self,
        chunk_id: int,
        chunk_events: list[SpeakerEvent],
        chunk_start_ms: float,
        chunk_end_ms: float,
    ) -> list[int]:
        """
        Store links between chunks and events with timing information.

        Args:
            chunk_id: ID of the chunk
            chunk_events: List of speaker events in this chunk
            chunk_start_ms: Start time of chunk in milliseconds
            chunk_end_ms: End time of chunk in milliseconds

        Returns:
            link_ids: List of link IDs that were inserted
        """
        link_ids = []
        chunk_start_sec = chunk_start_ms / 1000.0
        chunk_end_sec = chunk_end_ms / 1000.0

        with self.db.get_cursor() as cursor:
            for event in chunk_events:
                # Calculate event timing relative to chunk
                event_start_in_chunk = max(0.0, event.start - chunk_start_sec)
                event_end_in_chunk = min(
                    chunk_end_sec - chunk_start_sec, event.end - chunk_start_sec
                )
                event_duration_in_chunk = event_end_in_chunk - event_start_in_chunk

                # Find the corresponding event in the database
                cursor.execute(
                    """
                    SELECT id FROM audio_events 
                    WHERE start_time_sec = %s AND end_time_sec = %s AND speaker_label = %s
                    LIMIT 1
                """,
                    (event.start, event.end, event.speaker),
                )

                result = cursor.fetchone()
                if result:
                    event_id = result["id"]

                    cursor.execute(
                        """
                        INSERT INTO chunk_event_links 
                        (chunk_id, event_id, event_start_in_chunk_sec, event_end_in_chunk_sec, 
                         event_duration_in_chunk_sec)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """,
                        (
                            chunk_id,
                            event_id,
                            event_start_in_chunk,
                            event_end_in_chunk,
                            event_duration_in_chunk,
                        ),
                    )

                    link_id = cursor.fetchone()["id"]
                    link_ids.append(link_id)

        log.debug(f"Stored {len(link_ids)} chunk-event links for chunk {chunk_id}")
        return link_ids

    def store_processed_file(
        self,
        processed_file: ProcessedAudioFile,
        speaker_embeddings: dict[str, np.ndarray] | None = None,
    ) -> int:
        """
        Store all data from a processed audio file.

        Args:
            processed_file: ProcessedAudioFile object
            speaker_embeddings: Optional speaker embeddings dictionary

        Returns:
            audio_file_id: The ID of the stored audio file
        """
        audio_file_id = self.store_audio_file(processed_file)

        speaker_id_map = {}
        if speaker_embeddings:
            speaker_id_map = self.store_speaker_embeddings(speaker_embeddings)

        all_speaker_events = []
        for chunk in processed_file.chunks:
            all_speaker_events.extend(chunk.audio_chunk.speaker_events)

        # Remove duplicates based on start, end, and speaker
        unique_events = {}
        for event in all_speaker_events:
            key = (event.start, event.end, event.speaker)
            if key not in unique_events:
                unique_events[key] = event

        self.store_audio_events(
            audio_file_id, list(unique_events.values()), speaker_id_map
        )

        # Store chunks
        chunk_ids = self.store_audio_chunks(audio_file_id, processed_file.chunks)

        # Store chunk-event links
        for i, chunk in enumerate(processed_file.chunks):
            self.store_chunk_event_links(
                chunk_ids[i],
                chunk.audio_chunk.speaker_events,
                chunk.audio_chunk.start,
                chunk.audio_chunk.end,
            )

        log.info(
            f"Successfully stored all data for audio file {processed_file.source_file}"
        )
        return audio_file_id

    def get_speaker_duration_in_chunk(self, chunk_id: int, speaker_id: int) -> float:
        """
        Calculate how much a specific speaker (by ID) spoke in a chunk.

        Args:
            chunk_id: ID of the chunk
            speaker_id: Speaker ID to calculate duration for

        Returns:
            Total duration in seconds that the speaker spoke in this chunk
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT SUM(cel.event_duration_in_chunk_sec) as total_duration
                FROM chunk_event_links cel
                JOIN audio_events ae ON cel.event_id = ae.id
                WHERE cel.chunk_id = %s AND ae.speaker_id = %s
            """,
                (chunk_id, speaker_id),
            )

            result = cursor.fetchone()
            return result["total_duration"] if result["total_duration"] else 0.0

    def get_chunk_speaker_summary(self, chunk_id: int) -> list[dict]:
        """
        Get a summary of all speakers and their durations in a chunk.

        Args:
            chunk_id: ID of the chunk

        Returns:
            List of dictionaries with speaker_id and duration_sec
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT ae.speaker_id, SUM(cel.event_duration_in_chunk_sec) as duration_sec
                FROM chunk_event_links cel
                JOIN audio_events ae ON cel.event_id = ae.id
                WHERE cel.chunk_id = %s AND ae.speaker_id IS NOT NULL
                GROUP BY ae.speaker_id
                ORDER BY duration_sec DESC
            """,
                (chunk_id,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_speaker_similarities(self) -> list[dict]:
        """
        Get all speaker similarities using database's vector similarity operator.

        Returns:
            List of dictionaries with speaker1_id, speaker2_id, and similarity
        """
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    s1.id as speaker1_id,
                    s2.id as speaker2_id,
                    1 - (s1.embedding <=> s2.embedding) as similarity
                FROM speakers s1
                CROSS JOIN speakers s2
                WHERE s1.id < s2.id
                ORDER BY similarity DESC
            """)

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "speaker1_id": row["speaker1_id"],  # type: ignore
                        "speaker2_id": row["speaker2_id"],  # type: ignore
                        "similarity": float(row["similarity"]),  # type: ignore
                    }
                )

            return results
