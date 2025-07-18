import re
import statistics
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa
import whisper_timestamped as whisper


class Transcriber:
    def __init__(self, checkpoint_path: str, device: torch.device | None = None):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            checkpoint_path, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

    @torch.no_grad()
    @torch.inference_mode()
    def transcribe(self, audio_path: str) -> str:
        target_sr = 16000
        audio, sr = librosa.load(audio_path, sr=target_sr)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=False
        )
        return transcription[0]

    @torch.no_grad()
    @torch.inference_mode()
    def transcribe_with_confidence(self, audio_path: str) -> dict:
        target_sr = 16000
        audio, sr = librosa.load(audio_path, sr=target_sr)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        result = self.model.generate(
            input_features,
            do_sample=False,
            num_beams=5,
            return_dict_in_generate=True,
        )

        logp = self.model.compute_transition_scores(
            result.sequences, result.scores, normalize_logits=True
        )
        seq_conf = torch.exp(logp.mean())

        # Decode the generated tokens to get the actual text
        transcription = self.processor.batch_decode(
            result.sequences, skip_special_tokens=False
        )[0]
        # I don't understand but with batch_decode and return_dict_in_generate=True the transcript contains all special tokens
        clean_text = re.sub(r"<\|.*?\|>", "", transcription).strip()

        return {"text": clean_text, "confidence": float(seq_conf)}


class WhisperTimestampedTranscriber:
    def __init__(
        self, model_size: str = "openai/whisper-large-v3-turbo", device: str = "cuda"
    ):
        self.model = whisper.load_model(model_size, device=device)
        self.device = device

    def transcribe(self, audio_path: str) -> str:
        result = whisper.transcribe(
            self.model, audio_path, language="nl", verbose=False
        )
        return result["text"]

    def transcribe_with_timestamps(self, audio_path: str) -> dict:
        result = whisper.transcribe(
            self.model, audio_path, language="nl", verbose=False
        )
        # Add general confidence
        confidence = statistics.mean([c["confidence"] for c in result["segments"]])
        result["confidence"] = confidence
        return result
