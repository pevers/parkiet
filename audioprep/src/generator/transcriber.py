import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import whisper_timestamped as whisper
from whisper_timestamped.transcribe import transcribe_timestamped

from generator.schemas import AudioChunk, ProcessedAudioData, SpeakerEvent
from utils.audio import get_audio_duration


class ChunkTranscriber:
    def __init__(self, model_name: str, device: str = "cuda", language: str = "nl"):
        self.model = whisper.load_model(model_name, device=device)
        self.language = language
        self.device = device

    def transcribe_chunk(self, chunk_path: Path) -> dict:
        """
        Transcribe a single audio chunk.
        
        Args:
            chunk_path: Path to the audio chunk file
            
        Returns:
            Whisper transcription result with timestamps
        """
        result = transcribe_timestamped(
            self.model, 
            chunk_path.as_posix(),
            language=self.language,
            vad="auditok"
        )

        # Sanity check, get the length of the chunk and clip to that
        # Helps against Whisper hallucinating
        chunk_duration = get_audio_duration(chunk_path)
        result["segments"] = [segment for segment in result["segments"] if segment["end"] <= chunk_duration]

        return result

    def map_speakers_to_tags(self, speaker_events: list[SpeakerEvent]) -> dict[str, str]:
        """
        Map original speaker IDs to simplified tags (S1, S2, etc.) for a chunk.
        The first speaker to speak chronologically becomes S1.
        
        Args:
            speaker_events: List of speaker events in the chunk
            
        Returns:
            Dictionary mapping original speaker IDs to simplified tags
        """
        if not speaker_events:
            return {}
        
        # Sort events by start time to find chronological order
        sorted_events = sorted(speaker_events, key=lambda x: x.start)
        
        # Get unique speakers in chronological order (first appearance)
        unique_speakers = []
        seen_speakers = set()
        for event in sorted_events:
            if event.speaker not in seen_speakers:
                unique_speakers.append(event.speaker)
                seen_speakers.add(event.speaker)
        
        # Map speakers to tags, with first speaker chronologically being S1
        speaker_map = {}
        for i, speaker in enumerate(unique_speakers):
            speaker_map[speaker] = f"S{i + 1}"
        
        return speaker_map

    def combine_transcription_with_speakers(
        self, 
        transcription: dict, 
        speaker_events: list[SpeakerEvent],
        chunk_start_ms: float
    ) -> str:
        """
        Combine transcription with speaker events to create speaker-tagged text.
        
        Args:
            transcription: Whisper transcription result
            speaker_events: List of speaker events for this chunk
            chunk_start_ms: Start time of the chunk in milliseconds
            
        Returns:
            Text with speaker tags like "[S1] Hello [S2] Hi there"
        """
        if not transcription.get("segments") or not speaker_events:
            return ""
        
        # Convert chunk start from milliseconds to seconds for comparison
        chunk_start_sec = chunk_start_ms / 1000.0
        
        # Map speakers to simplified tags
        speaker_map = self.map_speakers_to_tags(speaker_events)
        
        # Convert speaker events to be relative to chunk start
        relative_speaker_events = []
        for event in speaker_events:
            relative_start = event.start - chunk_start_sec
            relative_end = event.end - chunk_start_sec
            relative_speaker_events.append({
                'start': max(0, relative_start),  # Ensure non-negative
                'end': relative_end,
                'speaker': speaker_map[event.speaker]
            })
        
        # Sort speaker events by start time
        relative_speaker_events.sort(key=lambda x: x['start'])
        
        # Collect all words from all segments with their timestamps
        all_words = []
        for segment in transcription["segments"]:
            if "words" in segment:
                for word in segment["words"]:
                    all_words.append({
                        'text': word['text'],
                        'start': word['start'],
                        'end': word['end']
                    })
        
        if not all_words:
            return ""
        
        # Sort words by start time
        all_words.sort(key=lambda x: x['start'])
        
        # Find speaker transitions - when the speaker actually changes
        speaker_transitions = []
        current_speaker = None
        
        for event in relative_speaker_events:
            if event['speaker'] != current_speaker:
                speaker_transitions.append({
                    'time': event['start'],
                    'speaker': event['speaker']
                })
                current_speaker = event['speaker']
        
        # Sort transitions by time (should already be sorted, but just to be safe)
        speaker_transitions.sort(key=lambda x: x['time'])
        
        # For each transition, find the closest word end after the transition time
        transition_placements = []
        for transition in speaker_transitions:
            transition_time = transition['time']
            speaker = transition['speaker']
            
            # Find the first word that ends after the transition time
            best_word_idx = None
            for i, word in enumerate(all_words):
                if word['end'] >= transition_time:
                    best_word_idx = i
                    break
            
            if best_word_idx is not None:
                transition_placements.append({
                    'word_idx': best_word_idx,
                    'speaker': speaker,
                    'transition_time': transition_time
                })
        
        # Sort placements by word index to process in order
        transition_placements.sort(key=lambda x: x['word_idx'])
        
        # Build result with speaker tags placed before appropriate words
        result_parts = []
        transition_idx = 0
        last_speaker_added = None
        
        # Add first speaker tag at the beginning if we have transitions
        if speaker_transitions:
            first_speaker = speaker_transitions[0]['speaker']
            result_parts.append(f"[{first_speaker}]")
            last_speaker_added = first_speaker
        
        for word_idx, word in enumerate(all_words):            
            # Check if we need to add a speaker transition after this word
            while (transition_idx < len(transition_placements) and 
                   transition_placements[transition_idx]['word_idx'] == word_idx):
                
                placement = transition_placements[transition_idx]
                # Only add speaker tag if it's different from the last one we added
                if placement['speaker'] != last_speaker_added:
                    result_parts.append(f"[{placement['speaker']}]")
                    last_speaker_added = placement['speaker']
                
                transition_idx += 1
            result_parts.append(word['text'])
        
        return " ".join(result_parts)

    def process_chunk(self, chunk: AudioChunk, chunk_dir: Path) -> dict:
        """
        Process a single chunk: transcribe and add speaker tags.
        
        Args:
            chunk: AudioChunk object
            chunk_dir: Directory containing the chunk files
            
        Returns:
            Dictionary with transcription results
        """
        chunk_path = chunk_dir / chunk.file_path
        
        if not chunk_path.exists():
            return {
                'chunk_id': chunk.file_path,
                'success': False,
                'error': f'Chunk file not found: {chunk_path}',
                'transcription': '',
                'speaker_tagged_text': ''
            }
        
        try:
            print(f"Transcribing chunk: {chunk.file_path}")
            
            # Transcribe the chunk
            transcription = self.transcribe_chunk(chunk_path)
            
            # Combine with speaker information
            speaker_tagged_text = self.combine_transcription_with_speakers(
                transcription, chunk.speaker_events, chunk.start
            )
            
            return {
                'chunk_id': chunk.file_path,
                'success': True,
                'transcription': transcription,
                'speaker_tagged_text': speaker_tagged_text,
                'start_ms': chunk.start,
                'end_ms': chunk.end,
                'speaker_events': [
                    {
                        'start': event.start,
                        'end': event.end,
                        'speaker': event.speaker
                    } for event in chunk.speaker_events
                ]
            }
            
        except Exception as e:
            return {
                'chunk_id': chunk.file_path,
                'success': False,
                'error': str(e),
                'transcription': '',
                'speaker_tagged_text': ''
            }


def process_chunks_folder(
    chunks_folder: str, 
    model_name: str = "whisper-large-v3", 
    device: str = "cuda",
    language: str = "nl",
    max_workers: int = 1
) -> None:
    """
    Process all chunks in a folder by transcribing them with speaker tags.
    
    Args:
        chunks_folder: Path to folder containing processed chunks
        model_name: Whisper model name to use
        device: Device to run on (cpu/cuda)
        language: Language code for transcription
        max_workers: Maximum number of worker threads
    """
    chunks_path = Path(chunks_folder)
    
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks folder not found: {chunks_path}")
    
    # Find all subdirectories (each represents a processed audio file)
    audio_dirs = [d for d in chunks_path.iterdir() if d.is_dir()]
    
    if not audio_dirs:
        print(f"No audio directories found in: {chunks_path}")
        return
    
    print(f"Found {len(audio_dirs)} audio directories to process")
    
    # Initialize transcriber
    transcriber = ChunkTranscriber(model_name=model_name, device=device, language=language)    
    for audio_dir in audio_dirs:
        print(f"\nProcessing audio directory: {audio_dir.name}")
        
        # Load processed_data.json
        processed_data_path = audio_dir / "processed_data.json"
        if not processed_data_path.exists():
            print(f"No processed_data.json found in {audio_dir}, skipping")
            continue
        
        with open(processed_data_path, 'r') as f:
            processed_data_dict = json.load(f)
        
        # Convert to ProcessedAudioData object
        processed_data = ProcessedAudioData(**processed_data_dict)
        
        if not processed_data.chunks:
            print(f"No chunks found in {audio_dir}, skipping")
            continue
        
        print(f"Processing {len(processed_data.chunks)} chunks...")
        
        # Process chunks in parallel
        chunk_results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(transcriber.process_chunk, chunk, audio_dir): chunk
                for chunk in processed_data.chunks
            }
            
            for future in as_completed(future_to_chunk):
                result = future.result()
                chunk_results.append(result)
                
                if result['success']:
                    print(f"✓ Completed: {result['chunk_id']}")
                else:
                    print(f"✗ Failed: {result['chunk_id']} - {result.get('error', 'Unknown error')}")
        
        # Save results
        transcription_results = {
            'source_file': processed_data.source_file,
            'total_chunks': len(processed_data.chunks),
            'successful_chunks': len([r for r in chunk_results if r['success']]),
            'failed_chunks': len([r for r in chunk_results if not r['success']]),
            'processing_time_sec': time.time() - start_time,
            'chunks': chunk_results
        }
        
        # Save transcription results
        transcription_path = audio_dir / "transcription_results.json"
        with open(transcription_path, 'w') as f:
            json.dump(transcription_results, f, indent=2, ensure_ascii=False)
        
        # Create a simple text file with all speaker-tagged transcriptions
        text_output_path = audio_dir / "speaker_tagged_transcription.txt"
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(f"Speaker-tagged transcription for: {processed_data.source_file}\n")
            f.write("=" * 80 + "\n\n")
            
            successful_chunks = [r for r in chunk_results if r['success']]
            successful_chunks.sort(key=lambda x: x['start_ms'])
            
            for result in successful_chunks:
                if result['speaker_tagged_text'].strip():
                    f.write(f"Chunk {result['chunk_id']} ({result['start_ms']:.0f}ms - {result['end_ms']:.0f}ms):\n")
                    f.write(f"{result['speaker_tagged_text']}\n\n")
        
        print(f"Completed {audio_dir.name}:")
        print(f"  - Successful chunks: {transcription_results['successful_chunks']}")
        print(f"  - Failed chunks: {transcription_results['failed_chunks']}")
        print(f"  - Processing time: {transcription_results['processing_time_sec']:.1f}s")
        print(f"  - Results saved to: {transcription_path}")
        print(f"  - Text output saved to: {text_output_path}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio chunks with speaker tagging using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/chunks
  %(prog)s /path/to/chunks --model ../data/whisper-large-v3-dutch-cgn-prod/
  %(prog)s /path/to/chunks --model whisper-large-v3 --device cuda --language nl --workers 2
        """
    )
    
    parser.add_argument(
        'chunks_folder',
        help='Path to folder containing processed audio chunks'
    )
    
    parser.add_argument(
        '--model', '-m',
        default='whisper-large-v3',
        help='Whisper model name to use (default: whisper-large-v3)'
    )
    
    parser.add_argument(
        '--device', '-d',
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Device to run on: cpu or cuda (default: cuda)'
    )
    
    parser.add_argument(
        '--language', '-l',
        default='nl',
        help='Language code for transcription (default: nl)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=1,
        help='Maximum number of worker threads'
    )
    
    args = parser.parse_args()
    process_chunks_folder(
        chunks_folder=args.chunks_folder,
        model_name=args.model,
        device=args.device,
        language=args.language,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
