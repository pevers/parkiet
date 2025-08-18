import logging
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

    def _find_similar_speaker(self, embedding: list[float]) -> int | None:
        with self.db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, 1 - (embedding <=> %s::vector) as similarity
                FROM speakers
                WHERE 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT 1
            """,
                (
                    embedding,
                    embedding,
                    self.similarity_threshold,
                    embedding,
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
        self, speaker_embeddings: dict[str, list[float]]
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
                    cursor.execute(
                        """
                        INSERT INTO speakers (embedding)
                        VALUES (%s::vector)
                        RETURNING id
                    """,
                        (embedding,),
                    )

                    speaker_id = cursor.fetchone()["id"]  # type: ignore
                    speaker_id_map[speaker_label] = speaker_id
                    log.info(f"Stored new speaker {speaker_label} with ID {speaker_id}")

        return speaker_id_map

    def store_audio_events(
        self,
        audio_file_id: int,
        chunk_id: int,
        speaker_events: list[SpeakerEvent],
        speaker_id_map: dict[str, int] | None = None,
    ) -> list[int]:
        """
        Store audio events in the database.

        Args:
            audio_file_id: ID of the audio file
            chunk_id: ID of the chunk these events belong to
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
                    (audio_file_id, chunk_id, speaker_id, start_time_sec, end_time_sec, speaker_label)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (
                        audio_file_id,
                        chunk_id,
                        speaker_id,
                        event.start,
                        event.end,
                        event.speaker,
                    ),
                )

                event_id = cursor.fetchone()["id"]  # type: ignore
                event_ids.append(event_id)

        log.info(f"Stored {len(event_ids)} audio events for chunk {chunk_id}")
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
                    (audio_file_id, chunk_file_path, start_time_ms, end_time_ms, transcription, transcription_clean)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (
                        audio_file_id,
                        chunk.audio_chunk.gcs_file_path or chunk.audio_chunk.file_path,
                        chunk.audio_chunk.start,
                        chunk.audio_chunk.end,
                        chunk.transcription,
                        chunk.transcription_clean,
                    ),
                )

                chunk_id = cursor.fetchone()["id"]  # type: ignore
                chunk_ids.append(chunk_id)

        log.info(f"Stored {len(chunk_ids)} audio chunks for audio file {audio_file_id}")
        return chunk_ids

    def store_processed_file(
        self,
        processed_file: ProcessedAudioFile,
        speaker_embeddings: dict[str, list[float]] | None = None,
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

        # Store chunks if any exist
        if processed_file.chunks:
            chunk_ids = self.store_audio_chunks(audio_file_id, processed_file.chunks)

            # Store events for each chunk
            for i, chunk in enumerate(processed_file.chunks):
                chunk_events = chunk.audio_chunk.speaker_events
                self.store_audio_events(
                    audio_file_id, chunk_ids[i], chunk_events, speaker_id_map
                )

        log.info(
            f"Successfully stored audio file {processed_file.source_file} with ID: {audio_file_id}"
        )
        return audio_file_id

    def add_chunks_to_audio_file(
        self,
        audio_file_id: int,
        processed_chunks: list[ProcessedAudioChunk],
        speaker_embeddings: dict[str, list[float]] | None = None,
    ) -> list[int]:
        """
        Add chunks to an existing audio file.

        Args:
            audio_file_id: ID of the existing audio file
            processed_chunks: List of processed audio chunks to add
            speaker_embeddings: Optional speaker embeddings dictionary

        Returns:
            chunk_ids: List of chunk IDs that were inserted
        """
        speaker_id_map = {}
        if speaker_embeddings:
            speaker_id_map = self.store_speaker_embeddings(speaker_embeddings)

        # Store chunks
        chunk_ids = self.store_audio_chunks(audio_file_id, processed_chunks)

        # Store events for each chunk
        for i, chunk in enumerate(processed_chunks):
            chunk_events = chunk.audio_chunk.speaker_events
            self.store_audio_events(
                audio_file_id, chunk_ids[i], chunk_events, speaker_id_map
            )

        log.info(
            f"Successfully added {len(processed_chunks)} chunks to audio file {audio_file_id}"
        )
        return chunk_ids

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
                SELECT SUM(end_time_sec - start_time_sec) as total_duration
                FROM audio_events
                WHERE chunk_id = %s AND speaker_id = %s
            """,
                (chunk_id, speaker_id),
            )

            result = cursor.fetchone()
            return result["total_duration"] if result["total_duration"] else 0.0  # type: ignore

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
                SELECT speaker_id, SUM(end_time_sec - start_time_sec) as duration_sec
                FROM audio_events
                WHERE chunk_id = %s AND speaker_id IS NOT NULL
                GROUP BY speaker_id
                ORDER BY duration_sec DESC
            """,
                (chunk_id,),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_chunk_events(self, chunk_id: int) -> list[dict]:
        """
        Get all audio events for a specific chunk.

        Args:
            chunk_id: ID of the chunk

        Returns:
            List of dictionaries with event details
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT id, speaker_id, speaker_label, start_time_sec, end_time_sec
                FROM audio_events
                WHERE chunk_id = %s
                ORDER BY start_time_sec
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

    def get_all_audio_chunks(self, limit: int | None = None) -> list[dict]:
        """
        Get all audio chunks from the database.

        Args:
            limit: Optional limit on number of chunks to return

        Returns:
            List of dictionaries with chunk data including transcriptions
        """
        with self.db.get_cursor() as cursor:
            query = """
                SELECT 
                    ac.id as chunk_id,
                    ac.chunk_file_path,
                    ac.start_time_ms,
                    ac.end_time_ms,
                    ac.transcription,
                    ac.transcription_clean,
                    af.original_file_path,
                    af.audio_duration_sec
                FROM audio_chunks ac
                JOIN audio_files af ON ac.audio_file_id = af.id
                WHERE af.success = true
                ORDER BY af.id, ac.start_time_ms
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def calculate_speaker_weights(self) -> dict[int, float]:
        """
        Calculate weights for all speakers based on their total speaking time.
        Weight is normalized so all weights sum to 1.0.

        Returns:
            Dictionary mapping speaker_id to weight (0.0-1.0)
        """
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    speaker_id,
                    SUM(end_time_sec - start_time_sec) as total_duration
                FROM audio_events
                WHERE speaker_id IS NOT NULL
                GROUP BY speaker_id
            """)

            results = cursor.fetchall()
            if not results:
                return {}

            # Calculate total duration across all speakers
            total_duration = sum(row["total_duration"] for row in results)

            # Calculate normalized weights
            weights = {}
            for row in results:
                speaker_id = row["speaker_id"]
                duration = row["total_duration"]
                weights[speaker_id] = (
                    float(duration / total_duration) if total_duration > 0 else 0.0
                )

            return weights

    def get_chunk_speaker_weight(
        self, chunk_id: int, speaker_weights: dict[int, float]
    ) -> float:
        """
        Calculate the weight of a chunk based on the speakers present and their weights.

        Args:
            chunk_id: ID of the chunk
            speaker_weights: Dictionary mapping speaker_id to weight

        Returns:
            Combined weight for the chunk
        """
        with self.db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    speaker_id,
                    SUM(end_time_sec - start_time_sec) as duration_in_chunk
                FROM audio_events
                WHERE chunk_id = %s AND speaker_id IS NOT NULL
                GROUP BY speaker_id
            """,
                (chunk_id,),
            )

            chunk_events = cursor.fetchall()
            if not chunk_events:
                return 0.0

            # Calculate total duration of this chunk
            total_chunk_duration = sum(
                event["duration_in_chunk"] for event in chunk_events
            )

            # Calculate weighted average based on speaker presence
            weighted_sum = 0.0
            for event in chunk_events:
                speaker_id = event["speaker_id"]
                speaker_duration = event["duration_in_chunk"]
                speaker_weight = speaker_weights.get(speaker_id, 0.0)

                # Weight by proportion of chunk this speaker occupies
                proportion = (
                    speaker_duration / total_chunk_duration
                    if total_chunk_duration > 0
                    else 0.0
                )
                weighted_sum += speaker_weight * proportion

            return weighted_sum
