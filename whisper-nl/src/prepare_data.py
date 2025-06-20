"""
CGN Data Preparation Script for Whisper Fine-tuning (Clean Version)

This script processes the CGN (Corpus Gesproken Nederlands) data and filters out
inaudible markers ("xxx") to create cleaner training data
"""

import gzip
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import html
from utils.audio import get_audio_duration, extract_audio_chunk

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "laughs": "(laughs)",
}


class CGNProcessorClean:
    """Processes CGN corpus data for Whisper fine-tuning with cleaned transcripts"""

    def __init__(
        self,
        filter_inaudible: bool = True,
        max_inaudible_ratio: float = 0.3,
        base_dir: str = "../data/CGN_2.0.3",
    ):
        self.base_dir = Path(base_dir)
        self.audio_dir = self.base_dir / "data/audio/wav"
        self.annot_dir = self.base_dir / "data/annot/xml/skp-ort"
        self.pri_dir = (
            self.base_dir / "data/annot/xml/pri"
        )  # Add PRI directory for punctuation
        self.output_dir = Path("../data/training")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Audio processing settings
        self.max_chunk_duration = 30.0  # seconds
        self.sample_rate = 16000
        self.channels = 1

        # Text cleaning settings
        self.filter_inaudible = filter_inaudible
        self.max_inaudible_ratio = (
            max_inaudible_ratio  # Max ratio of inaudible words per chunk
        )

        # Components to process (all CGN components a-o)
        self.components = [f"comp-{chr(ord('a') + i)}" for i in range(15)]
        self.regions = ["nl", "vl"]  # Dutch and Flemish

        # Statistics
        self.stats = {
            "total_chunks_before_filtering": 0,
            "chunks_with_inaudible": 0,
            "chunks_filtered_out": 0,
            "total_inaudible_words": 0,
            "total_laughter_words": 0,
            "total_words": 0,
            "punctuation_added": 0,
        }

        logger.info(f"Initialized CGN processor")
        logger.info(f"Filter inaudible: {self.filter_inaudible}")
        logger.info(f"Max inaudible ratio: {self.max_inaudible_ratio}")
        logger.info(f"Audio dir: {self.audio_dir}")
        logger.info(f"Annotation dir: {self.annot_dir}")
        logger.info(f"PRI dir: {self.pri_dir}")
        logger.info(f"Output dir: {self.output_dir}")

    def is_inaudible_word(self, word: str) -> bool:
        """Check if a word represents inaudible speech"""
        if not word or not word.strip():
            return False

        word_lower = word.lower().strip()

        # Check for CGN inaudible markers (but NOT laughter)
        inaudible_markers = [
            "xxx",  # Standard inaudible marker
            "xxxx",  # Longer inaudible marker
            "*",  # Sometimes used for unclear speech
        ]

        return word_lower in inaudible_markers

    def is_laughter_word(self, word: str) -> bool:
        """Check if a word represents laughter"""
        if not word or not word.strip():
            return False

        word_lower = word.lower().strip()

        # Check for CGN laughter markers
        laughter_markers = [
            "ggg",  # Standard laughter marker
            "gggg",  # Longer laughter
        ]

        return word_lower in laughter_markers

    def clean_word(self, word: str) -> str | None:
        """Clean a word, returning None if it should be filtered out"""
        if not word or not word.strip():
            return None

        # Convert laughter to readable form
        if self.is_laughter_word(word):
            return SPECIAL_TOKENS["laughs"]

        # Remove if it's an inaudible marker
        if self.is_inaudible_word(word):
            return None

        # Basic cleaning - remove extra whitespace
        cleaned = word.strip()

        # Keep only non-empty words
        return cleaned if cleaned else None

    def _preprocess_xml_entities(self, content: str) -> str:
        """
        Preprocess XML content to handle HTML entities that XML parser doesn't recognize.
        This converts HTML entities to their Unicode equivalents while preserving XML structure.
        """
        html_entities = {
            "&eacute;": "é",
            "&Eacute;": "É",
            "&egrave;": "è",
            "&Egrave;": "È",
            "&ecirc;": "ê",
            "&Ecirc;": "Ê",
            "&euml;": "ë",
            "&Euml;": "Ë",
            "&aacute;": "á",
            "&Aacute;": "Á",
            "&agrave;": "à",
            "&Agrave;": "À",
            "&acirc;": "â",
            "&Acirc;": "Â",
            "&auml;": "ä",
            "&Auml;": "Ä",
            "&atilde;": "ã",
            "&Atilde;": "Ã",
            "&iacute;": "í",
            "&Iacute;": "Í",
            "&igrave;": "ì",
            "&Igrave;": "Ì",
            "&icirc;": "î",
            "&Icirc;": "Î",
            "&iuml;": "ï",
            "&Iuml;": "Ï",
            "&oacute;": "ó",
            "&Oacute;": "Ó",
            "&ograve;": "ò",
            "&Ograve;": "Ò",
            "&ocirc;": "ô",
            "&Ocirc;": "Ô",
            "&ouml;": "ö",
            "&Ouml;": "Ö",
            "&uacute;": "ú",
            "&Uacute;": "Ú",
            "&ugrave;": "ù",
            "&Ugrave;": "Ù",
            "&ucirc;": "û",
            "&Ucirc;": "Û",
            "&uuml;": "ü",
            "&Uuml;": "Ü",
            "&ccedil;": "ç",
            "&Ccedil;": "Ç",
            "&ntilde;": "ñ",
            "&Ntilde;": "Ñ",
            "&nbsp;": " ",
            "&ndash;": "–",
            "&mdash;": "—",
            "&lsquo;": """,
            '&rsquo;': """,
            "&ldquo;": '"',
            "&rdquo;": '"',
            "&hellip;": "…",
            "&trade;": "™",
            "&copy;": "©",
            "&reg;": "®",
            "&oslash;": "ø",
            "&Oslash;": "Ø",
            "&Aring;": "Å",
            "&aring;": "å",
        }

        # Replace HTML entities with Unicode equivalents
        for entity, replacement in html_entities.items():
            content = content.replace(entity, replacement)

        return content

    def clean_text(self, word_data_list: list[dict]) -> tuple[str, float]:
        """
        Clean text by removing inaudible markers and converting laughter, including punctuation.
        Insert speaker tags whenever the speaker changes, but remap them per chunk so the first speaker is [S1], the second [S2], etc.
        Returns: (cleaned_text, inaudible_ratio)
        """
        total_words = len(word_data_list)
        if total_words == 0:
            return "", 0.0

        # Remap speakers per chunk
        speaker_remap = {}
        remap_counter = 1
        for word_data in word_data_list:
            speaker = word_data.get("speaker", None)
            if speaker and speaker not in speaker_remap:
                speaker_remap[speaker] = f"[S{remap_counter}]"
                remap_counter += 1

        cleaned_parts = []
        inaudible_count = 0
        laughter_count = 0
        punctuation_added = 0

        last_speaker = None
        for word_data in word_data_list:
            word = (
                word_data.get("word", "") if isinstance(word_data, dict) else word_data
            )
            punctuation = (
                word_data.get("punctuation", "") if isinstance(word_data, dict) else ""
            )
            speaker = word_data.get("speaker", None)
            remapped_speaker = speaker_remap.get(speaker, speaker) if speaker else None

            # Insert remapped speaker tag if speaker changes or at the start
            if remapped_speaker and remapped_speaker != last_speaker:
                cleaned_parts.append(remapped_speaker)
                last_speaker = remapped_speaker

            if self.is_laughter_word(word):
                laughter_count += 1
                word_with_punct = SPECIAL_TOKENS["laughs"]
                if punctuation:
                    word_with_punct += punctuation
                    punctuation_added += 1
                cleaned_parts.append(word_with_punct)
            elif self.is_inaudible_word(word):
                inaudible_count += 1
                # Don't add the word, but still add punctuation if present
                if punctuation:
                    # Add punctuation to the previous word if it exists
                    if cleaned_parts:
                        cleaned_parts[-1] += punctuation
                        punctuation_added += 1
                continue
            else:
                cleaned_word = self.clean_word(word)
                if cleaned_word:
                    # Always attach punctuation directly to the word
                    if punctuation:
                        cleaned_word += punctuation
                        punctuation_added += 1
                    cleaned_parts.append(cleaned_word)

        # Update punctuation statistics
        self.stats["punctuation_added"] += punctuation_added

        inaudible_ratio = inaudible_count / total_words if total_words > 0 else 0
        cleaned_text = " ".join(cleaned_parts)

        return cleaned_text, inaudible_ratio

    def parse_pri_file(self, pri_file: Path) -> dict[str, str]:
        """Parse PRI file to extract punctuation marks mapped by word ID"""
        punctuation_map = {}

        with gzip.open(pri_file, "rt", encoding="utf-8") as f:
            content = f.read()

        # Parse XML first, then unescape individual text values
        content = self._preprocess_xml_entities(content)
        root = ET.fromstring(content)

        # Extract punctuation marks from <l> tags
        for l_tag in root.findall(".//l"):
            punct_id = l_tag.get("id", "")
            punct_text = l_tag.text.strip() if l_tag.text else ""

            if punct_id and punct_text:
                # Decode HTML entities in punctuation text
                punct_text = html.unescape(punct_text)

                # Extract the word position from punctuation ID
                # e.g., "fn000248.1.19" -> "fn000248.1.18" (previous word)
                parts = punct_id.split(".")
                if len(parts) >= 3:
                    try:
                        word_num = (
                            int(parts[-1]) - 1
                        )  # Punctuation follows previous word
                        if word_num > 0:
                            word_id = ".".join(parts[:-1] + [str(word_num)])
                            punctuation_map[word_id] = punct_text
                    except ValueError:
                        continue

        return punctuation_map

    def parse_xml_annotation(self, xml_file: Path, pri_file: Path = None) -> list[dict]:
        """Parse CGN XML annotation file and extract word-level timestamps"""

        with gzip.open(xml_file, "rt", encoding="utf-8") as f:
            content = f.read()

        # Handle HTML entities that XML parser doesn't recognize
        content = self._preprocess_xml_entities(content)
        # Parse XML first, then unescape individual text values
        root = ET.fromstring(content)
        words = []

        # Parse punctuation from PRI file if available
        punctuation_map = {}
        if pri_file and pri_file.exists():
            punctuation_map = self.parse_pri_file(pri_file)

        speaker_map = {}

        # Extract all word elements with timestamps
        for tau in root.findall(".//tau"):
            if tau.get("s", "") == "COMMENT":
                # Skip comments
                continue
            speaker = tau.get("s", "")
            if speaker not in speaker_map:
                speaker_map[speaker] = len(speaker_map)

            for tw in tau.findall(".//tw"):
                word_text = tw.get("w", "")
                word_id = tw.get("ref", "")

                # Skip empty words
                if not word_text.strip():
                    continue

                # Clean up whitespace and decode HTML entities in the word text
                word_text = html.unescape(word_text.strip())

                # Check if this word has punctuation following it
                punctuation = punctuation_map.get(word_id, "")

                word_data = {
                    "speaker": f"[S{speaker_map[speaker] + 1}]",
                    "word": word_text,
                    "start": float(tw.get("tb", 0)),
                    "end": float(tw.get("te", 0)),
                    "is_inaudible": self.is_inaudible_word(word_text),
                    "is_laughter": self.is_laughter_word(word_text),
                    "punctuation": punctuation,  # Add punctuation info
                    "word_id": word_id,  # Keep word ID for debugging
                }
                words.append(word_data)

        # Sort by start time
        words.sort(key=lambda x: x["start"])
        return words

    def create_chunks(self, words: list[dict]) -> list[dict]:
        """Create audio chunks respecting word boundaries and filtering inaudible content"""
        if not words:
            return []

        chunks = []
        current_chunk = {"start": 0.0, "end": 0.0, "words": [], "text": ""}

        for word in words:
            # Check if adding this word would exceed max duration
            potential_end = word["end"]
            potential_duration = potential_end - current_chunk["start"]

            if potential_duration > self.max_chunk_duration and current_chunk["words"]:
                # Finalize current chunk
                current_chunk["end"] = current_chunk["words"][-1]["end"]
                chunk = self._finalize_chunk(current_chunk)
                if chunk:  # Only add if chunk passes filtering
                    chunks.append(chunk)

                # Start new chunk
                current_chunk = {
                    "start": word["start"],
                    "end": word["end"],
                    "words": [word],
                    "text": "",
                }
            else:
                # Add word to current chunk
                if not current_chunk["words"]:
                    current_chunk["start"] = word["start"]
                current_chunk["words"].append(word)
                current_chunk["end"] = word["end"]

        # Add final chunk if it has words
        if current_chunk["words"]:
            chunk = self._finalize_chunk(current_chunk)
            if chunk:  # Only add if chunk passes filtering
                chunks.append(chunk)

        return chunks

    def _finalize_chunk(self, chunk_data: dict) -> dict | None:
        """Finalize a chunk by cleaning text and applying filters"""
        # Pass the full word data (including punctuation) to clean_text
        cleaned_text, inaudible_ratio = self.clean_text(chunk_data["words"])

        # Update statistics
        self.stats["total_chunks_before_filtering"] += 1
        self.stats["total_words"] += len(chunk_data["words"])
        self.stats["total_inaudible_words"] += sum(
            1 for w in chunk_data["words"] if w["is_inaudible"]
        )
        self.stats["total_laughter_words"] += sum(
            1 for w in chunk_data["words"] if w["is_laughter"]
        )

        if inaudible_ratio > 0:
            self.stats["chunks_with_inaudible"] += 1

        # Filter out chunk if it has too many inaudible words or is empty
        if self.filter_inaudible and (
            inaudible_ratio > self.max_inaudible_ratio or not cleaned_text.strip()
        ):
            self.stats["chunks_filtered_out"] += 1
            return None

        # Determine if there is a speaker change in the chunk
        speakers = [w.get("speaker") for w in chunk_data["words"] if w.get("speaker")]
        unique_speakers = set(speakers)
        speaker_change = len(unique_speakers) > 1

        # Track the largest speaker count per chunk
        speaker_count = len(unique_speakers)
        if "max_speaker_count" not in self.stats:
            self.stats["max_speaker_count"] = 0
        if speaker_count > self.stats["max_speaker_count"]:
            self.stats["max_speaker_count"] = speaker_count

        # Create final chunk (no 'speaker' field)
        final_chunk = {
            "start": chunk_data["start"],
            "end": chunk_data["end"],
            "text": cleaned_text,
            "inaudible_ratio": inaudible_ratio,
            "word_count": len(
                [w for w in chunk_data["words"] if not w["is_inaudible"]]
            ),
            "total_word_count": len(chunk_data["words"]),
            "laughter_count": len([w for w in chunk_data["words"] if w["is_laughter"]]),
            "speaker_change": speaker_change,
            "speaker_count": speaker_count,
        }

        return final_chunk

    def process_file_pair(
        self, audio_file: Path, annotation_file: Path, component: str, region: str
    ) -> list[dict] | None:
        """Process a single audio/annotation file pair"""
        file_id = audio_file.stem
        logger.info(f"Processing {component}/{region}/{file_id}")

        # Find corresponding PRI file for punctuation
        pri_file = self.pri_dir / component / region / f"{file_id}.pri.gz"

        # Parse annotation with punctuation
        words = self.parse_xml_annotation(annotation_file, pri_file)
        if not words:
            logger.warning(f"No words found in {annotation_file}")
            return []

        # Get audio duration
        audio_duration = get_audio_duration(audio_file)
        if not audio_duration:
            logger.warning(f"Could not get duration for {audio_file}")
            return []

        # Create chunks
        chunks = self.create_chunks(words)
        if not chunks:
            logger.warning(f"No valid chunks created for {file_id}")
            return []

        # Create output directories
        component_output_dir = self.output_dir / component / region
        component_output_dir.mkdir(parents=True, exist_ok=True)

        processed_chunks = []

        for i, chunk in enumerate(chunks):
            if not chunk["text"].strip():
                continue

            chunk_id = f"{file_id}_{i:03d}"
            chunk_audio_file = component_output_dir / f"{chunk_id}.wav"

            # Extract audio chunk
            if extract_audio_chunk(
                audio_file,
                chunk_audio_file,
                chunk["start"],
                chunk["end"],
                self.sample_rate,
                self.channels,
            ):
                chunk_data = {
                    "audio_file": str(chunk_audio_file.relative_to(self.output_dir)),
                    "text": chunk["text"],
                    "duration": chunk["end"] - chunk["start"],
                    "start_time": chunk["start"],
                    "end_time": chunk["end"],
                    "component": component,
                    "region": region,
                    "source_file": file_id,
                    "chunk_id": chunk_id,
                    "inaudible_ratio": chunk["inaudible_ratio"],
                    "word_count": chunk["word_count"],
                    "total_word_count": chunk["total_word_count"],
                    "laughter_count": chunk["laughter_count"],
                }

                processed_chunks.append(chunk_data)

            else:
                logger.error(f"Failed to extract audio for chunk {chunk_id}")

        return processed_chunks

    def find_file_pairs(self) -> list[tuple[Path, Path, str, str]] | None:
        """Find matching audio/annotation file pairs"""
        pairs = []

        for component in self.components:
            audio_comp_dir = self.audio_dir / component
            annot_comp_dir = self.annot_dir / component

            if not audio_comp_dir.exists() or not annot_comp_dir.exists():
                logger.warning(
                    f"Skipping {component} - directories not found {audio_comp_dir} or {annot_comp_dir}"
                )
                continue

            for region in self.regions:
                audio_region_dir = audio_comp_dir / region
                annot_region_dir = annot_comp_dir / region

                if not audio_region_dir.exists() or not annot_region_dir.exists():
                    logger.warning(
                        f"Skipping {component}/{region} - directories not found {audio_region_dir} or {annot_region_dir}"
                    )
                    continue

                # Find annotation files
                annot_files = list(annot_region_dir.glob("*.skp.gz"))

                for annot_file in annot_files:
                    # Extract file ID from annotation filename
                    file_id = annot_file.name.replace(".skp.gz", "")

                    # Find corresponding audio file
                    audio_file = audio_region_dir / f"{file_id}.wav"

                    if audio_file.exists():
                        pairs.append((audio_file, annot_file, component, region))
                    else:
                        logger.warning(f"No audio file found for {annot_file}")

        logger.info(f"Found {len(pairs)} audio/annotation pairs")
        return pairs

    def process_all_files(self, max_workers: int = 4):
        """Process all CGN files with parallel processing"""
        pairs = self.find_file_pairs()

        if not pairs:
            logger.error("No file pairs found to process")
            return

        all_chunks = []
        failed_files = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(
                    self.process_file_pair, audio_file, annot_file, component, region
                ): (audio_file, annot_file, component, region)
                for audio_file, annot_file, component, region in pairs
            }

            # Process completed tasks
            for future in tqdm(
                as_completed(future_to_pair), total=len(pairs), desc="Processing files"
            ):
                audio_file, annot_file, component, region = future_to_pair[future]

                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to process {audio_file}: {e}")
                    failed_files.append(str(audio_file))

        self.save_metadata(all_chunks, failed_files)
        self.print_statistics(all_chunks)
        logger.info(f"Processing complete!")
        logger.info(f"Total chunks created: {len(all_chunks)}")
        logger.info(f"Failed files: {len(failed_files)}")

        # Calculate statistics
        total_duration = sum(chunk["duration"] for chunk in all_chunks)
        logger.info(f"Total audio duration: {total_duration / 3600:.2f} hours")

    def print_statistics(self, final_chunks: list[dict] | None):
        """Print detailed statistics about the processing"""
        logger.info("=== Processing Statistics ===")
        logger.info(
            f"Total chunks before filtering: {self.stats['total_chunks_before_filtering']}"
        )
        logger.info(
            f"Chunks with inaudible words: {self.stats['chunks_with_inaudible']}"
        )
        logger.info(f"Chunks filtered out: {self.stats['chunks_filtered_out']}")
        logger.info(f"Final chunks: {len(final_chunks)}")

        if self.stats["total_chunks_before_filtering"] > 0:
            retention_rate = (
                len(final_chunks) / self.stats["total_chunks_before_filtering"] * 100
            )
            logger.info(f"Retention rate: {retention_rate:.1f}%")

        logger.info(f"Total words processed: {self.stats['total_words']}")
        logger.info(f"Inaudible words removed: {self.stats['total_inaudible_words']}")
        logger.info(f"Laughter words converted: {self.stats['total_laughter_words']}")
        logger.info(f"Punctuation marks added: {self.stats['punctuation_added']}")

        if self.stats["total_words"] > 0:
            inaudible_pct = (
                self.stats["total_inaudible_words"] / self.stats["total_words"] * 100
            )
            laughter_pct = (
                self.stats["total_laughter_words"] / self.stats["total_words"] * 100
            )
            punctuation_pct = (
                self.stats["punctuation_added"] / self.stats["total_words"] * 100
            )
            logger.info(f"Inaudible word ratio: {inaudible_pct:.1f}%")
            logger.info(f"Laughter word ratio: {laughter_pct:.1f}%")
            logger.info(f"Punctuation ratio: {punctuation_pct:.1f}%")

        if final_chunks:
            avg_inaudible_ratio = sum(
                chunk["inaudible_ratio"] for chunk in final_chunks
            ) / len(final_chunks)
            avg_chunk_duration = sum(chunk["duration"] for chunk in final_chunks) / len(
                final_chunks
            )
            avg_words_per_chunk = sum(
                chunk["word_count"] for chunk in final_chunks
            ) / len(final_chunks)
            avg_laughter_per_chunk = sum(
                chunk["laughter_count"] for chunk in final_chunks
            ) / len(final_chunks)

            logger.info(
                f"Average inaudible ratio in kept chunks: {avg_inaudible_ratio * 100:.1f}%"
            )
            logger.info(f"Average chunk duration: {avg_chunk_duration:.1f}s")
            logger.info(f"Average words per chunk: {avg_words_per_chunk:.1f}")
            logger.info(
                f"Average laughter instances per chunk: {avg_laughter_per_chunk:.1f}"
            )

        # Print the largest speaker count found in any chunk
        if "max_speaker_count" in self.stats:
            logger.info(
                f"Largest speaker count in any chunk: {self.stats['max_speaker_count']}"
            )

    def save_metadata(self, chunks: list[dict] | None, failed_files: list[str] | None):
        """Save processing metadata and create Whisper-compatible dataset files"""

        # Save complete metadata with statistics
        metadata = {
            "chunks": chunks,
            "failed_files": failed_files,
            "total_chunks": len(chunks),
            "total_duration": sum(chunk["duration"] for chunk in chunks),
            "components_processed": list(set(chunk["component"] for chunk in chunks)),
            "regions_processed": list(set(chunk["region"] for chunk in chunks)),
            "processing_stats": self.stats,
            "filter_settings": {
                "filter_inaudible": self.filter_inaudible,
                "max_inaudible_ratio": self.max_inaudible_ratio,
            },
        }

        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Create Whisper-compatible dataset file
        whisper_data = []
        for chunk in chunks:
            whisper_data.append({"audio": chunk["audio_file"], "text": chunk["text"]})

        whisper_file = self.output_dir / "whisper_dataset.json"
        with open(whisper_file, "w", encoding="utf-8") as f:
            json.dump(whisper_data, f, indent=2, ensure_ascii=False)

        # Create CSV for easier analysis
        csv_file = self.output_dir / "dataset.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write(
                "audio_file,text,duration,component,region,source_file,inaudible_ratio,word_count,laughter_count\n"
            )
            for chunk in chunks:
                text_escaped = chunk["text"].replace('"', '""')
                f.write(
                    f'"{chunk["audio_file"]}","{text_escaped}",{chunk["duration"]},"{chunk["component"]}","{chunk["region"]}","{chunk["source_file"]}",{chunk["inaudible_ratio"]},{chunk["word_count"]},{chunk["laughter_count"]}\n'
                )

        logger.info(f"Metadata saved to {metadata_file}")
        logger.info(f"Whisper dataset saved to {whisper_file}")
        logger.info(f"CSV dataset saved to {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process CGN data with inaudible filtering"
    )
    parser.add_argument(
        "--keep-inaudible",
        action="store_true",
        help="Keep chunks with inaudible markers instead of filtering them out",
    )
    parser.add_argument(
        "--max-inaudible-ratio",
        type=float,
        default=0.3,
        help="Maximum ratio of inaudible words per chunk (default: 0.3)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="../data/CGN_2.0.3",
        help="Base directory for CGN data (default: ../data/CGN_2.0.3)",
    )

    args = parser.parse_args()
    processor = CGNProcessorClean(
        filter_inaudible=not args.keep_inaudible,
        max_inaudible_ratio=args.max_inaudible_ratio,
        base_dir=args.base_dir,
    )
    processor.process_all_files(max_workers=args.workers)


if __name__ == "__main__":
    main()
