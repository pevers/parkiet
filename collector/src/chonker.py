from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import ulid
import numpy as np
import sys
import os
import json
import subprocess
from pathlib import Path
from schemas import Chunk, Segment

model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label


def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    audio_array, sampling_rate = librosa.load(
        audio_path, sr=feature_extractor.sampling_rate
    )

    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]

    return predicted_label


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <transcript_json> <output_dir>")
        sys.exit(1)

    transcript_json = sys.argv[1]
    output_dir = Path(sys.argv[2])
    file_stem = Path(transcript_json).stem
    chunk_dir = output_dir / file_stem
    chunk_dir.mkdir(parents=True, exist_ok=True)

    with open(transcript_json, "r") as f:
        transcript = json.load(f)

    # Infer audio file path from transcript JSON name
    audio_path = str(Path(transcript_json).with_suffix(".mp3"))
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)

    segments = []
    for utt in transcript.get("utterances", []):
        # Generate ULID filename
        audio_ulid = str(ulid.new())
        audio_filename = f"{audio_ulid}.mp3"
        audio_out_path = chunk_dir / audio_filename

        # Extract audio segment with ffmpeg (start/end in ms)
        start_sec = utt["start"] / 1000.0
        duration_sec = (utt["end"] - utt["start"]) / 1000.0
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            audio_path,
            "-ss",
            str(start_sec),
            "-t",
            str(duration_sec),
            "-ar",
            str(feature_extractor.sampling_rate),
            "-ac",
            "1",
            str(audio_out_path),
        ]
        subprocess.run(
            ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Run emotion recognition
        predicted_emotion = predict_emotion(
            str(audio_out_path), model, feature_extractor, id2label
        )

        # Build chunks from words
        chunks = [
            Chunk(
                speaker=w["speaker"],
                start=w["start"],
                end=w["end"],
                confidence=w["confidence"],
                text=w["text"],
            )
            for w in utt.get("words", [])
        ]

        segment = Segment(
            audio_url=audio_filename,
            start=utt["start"],
            end=utt["end"],
            confidence=utt["confidence"],
            text=utt["text"],
            chunks=chunks,
        )
        # Optionally, you could add the predicted emotion to the segment (as a new field)
        segment_dict = segment.model_dump()
        segment_dict["emotion"] = predicted_emotion
        segments.append(segment_dict)

    # Output JSON
    out_json_path = chunk_dir / "chunks.json"
    with open(out_json_path, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"Wrote segments to {out_json_path}")
