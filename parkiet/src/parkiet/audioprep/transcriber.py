from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa


class Transcriber:
    def __init__(self, checkpoint_path: str, device: torch.device | None = None):
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            checkpoint_path, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.generation_config.suppress_tokens = []
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
