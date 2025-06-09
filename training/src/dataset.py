import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torchaudio
from torch.utils.data import Dataset


class AudioPromptDataset(Dataset):
    """
    PyTorch dataset for loading audio samples and their corresponding prompts.
    Implements 15% prompt dropout as specified.
    """
    
    def __init__(
        self,
        audio_dir: Union[str, Path],
        prompts_json: Union[str, Path],
        prompt_dropout_rate: float = 0.15,
        sample_rate: int = 16000,
        max_audio_length: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            audio_dir: Directory containing audio files
            prompts_json: Path to JSON file with prompts (format: {"id": "prompt text"})
            prompt_dropout_rate: Probability of dropping out prompts (default: 0.15)
            sample_rate: Target sample rate for audio (default: 16000)
            max_audio_length: Maximum audio length in samples (None for no limit)
            seed: Random seed for reproducible prompt dropout
        """
        self.audio_dir = Path(audio_dir)
        self.prompts_json = Path(prompts_json)
        self.prompt_dropout_rate = prompt_dropout_rate
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        
        if seed is not None:
            random.seed(seed)
        
        # Load all available data samples
        self.samples = self._load_samples()
        
    def _load_samples(self) -> List[Dict]:
        """Load all audio-prompt pairs from the audio directory and prompts JSON."""
        samples: List[Dict] = []
        
        if not self.audio_dir.exists():
            raise ValueError(f"Audio directory not found: {self.audio_dir}")
        
        if not self.prompts_json.exists():
            raise ValueError(f"Prompts JSON file not found: {self.prompts_json}")
        
        # Load prompts from JSON file
        try:
            with open(self.prompts_json, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Error loading prompts JSON: {e}")
        
        # Find all audio files and match with prompts
        for audio_file in self.audio_dir.iterdir():
            if not audio_file.is_file() or not audio_file.suffix.lower() in ['.mp3', '.wav', '.flac', '.m4a']:
                continue
            
            # Get sample ID from filename (without extension)
            sample_id = audio_file.stem
            
            # Check if we have a prompt for this sample
            if sample_id in prompts_data:
                sample = {
                    'audio_path': str(audio_file),
                    'prompt_text': prompts_data[sample_id],
                    'sample_id': sample_id
                }
                samples.append(sample)
            else:
                print(f"Warning: No prompt found for audio file {audio_file.name}")
        
        print(f"Loaded {len(samples)} audio-prompt pairs")
        return samples
    

    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dict containing:
                - 'audio': torch.Tensor of shape (channels, time)
                - 'prompt': str (empty string if dropped out)
                - 'sample_id': str with the sample identifier
        """
        sample = self.samples[idx]
        
        # Load audio
        try:
            waveform, orig_sample_rate = torchaudio.load(sample['audio_path'])
            
            # Resample if necessary
            if orig_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sample_rate, 
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            
            # Truncate if necessary
            if self.max_audio_length is not None and waveform.shape[1] > self.max_audio_length:
                waveform = waveform[:, :self.max_audio_length]
            
        except Exception as e:
            print(f"Error loading audio {sample['audio_path']}: {e}")
            # Return empty audio tensor as fallback
            waveform = torch.zeros(1, self.sample_rate)  # 1 second of silence
        
        # Apply prompt dropout
        prompt_text = sample['prompt_text']
        if random.random() < self.prompt_dropout_rate:
            prompt_text = ""  # Drop out the prompt
        
        return {
            'audio': waveform,
            'prompt': prompt_text,
            'sample_id': sample['sample_id']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Custom collate function for batching audio samples with variable lengths.
    
    Args:
        batch: List of samples from __getitem__
        
    Returns:
        Batched data with padded audio tensors
    """
    # Separate components
    audio_list = [item['audio'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    sample_ids = [item['sample_id'] for item in batch]
    
    # Find maximum length for padding
    max_length = max(audio.shape[1] for audio in audio_list)
    batch_size = len(audio_list)
    num_channels = audio_list[0].shape[0]
    
    # Create padded tensor
    padded_audio = torch.zeros(batch_size, num_channels, max_length)
    audio_lengths = []
    
    for i, audio in enumerate(audio_list):
        length = audio.shape[1]
        padded_audio[i, :, :length] = audio
        audio_lengths.append(length)
    
    return {
        'audio': padded_audio,
        'audio_lengths': torch.tensor(audio_lengths),
        'prompts': prompts,
        'sample_ids': sample_ids
    }


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    dataset = AudioPromptDataset(
        audio_dir="../../collector/data/audio",
        prompts_json="../../collector/data/prompts.json",
        prompt_dropout_rate=0.15,
        sample_rate=16000,
        seed=42  # For reproducible results
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test a single sample
        sample = dataset[0]
        print(f"Audio shape: {sample['audio'].shape}")
        print(f"Prompt: '{sample['prompt']}'")
        print(f"Sample ID: {sample['sample_id']}")
        
        # Test dataloader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        batch = next(iter(dataloader))
        print(f"\nBatch audio shape: {batch['audio'].shape}")
        print(f"Batch audio lengths: {batch['audio_lengths']}")
        print(f"Batch prompts: {batch['prompts']}")
        print(f"Batch sample IDs: {batch['sample_ids']}") 