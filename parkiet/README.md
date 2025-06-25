# Training

Parkiet is a fine-tuned version of Parakeet on Dutch audio data and builds upon the excellent work of the [Dia authors](https://huggingface.co/nari-labs/Dia-1.6B) and research from [Parakeet](https://jordandarefsky.com/blog/2024/parakeet/).

## Setup

You can install dependencies with the following command. Use the `--extra cuda` flag to install the CUDA dependencies.
Use the `--extra tpu` flag to install the TPU dependencies.

```bash
uv sync --extra cuda
```

Download the weights:

```bash
wget https://huggingface.co/nari-labs/Dia-1.6B/resolve/main/dia-v0_1.pth?download=true -O weights/dia-v0_1.pth
```

## Prerequisites

```bash
sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
```

## JAX

The PyTorch model was ported to JAX to run it on Google Research TPUs.

## Data Collection Strategy

The `chunker` processes audio files, creates chunks, transcribes them, and adds them to Apache Arrow tables.

Quality control guidelines:
- Skip trailer episodes
- Skip episode introductions
- Skip non-Dutch content or server-side ads
- Skip segments with loud background music (using Pyannote)

To run the data pre-processing pipeline you need a Huggingface token because we use the pyannote speaker diarization model.

```bash
HG_TOKEN=<token> uv run python src/audioprep/chunker.py /path/to/source /path/to/target --workers 4
```

## Data Annotation

Using Label Studio for annotation with the following categories:

**Disfluencies:**
- Discourse Fillers: "zeg maar", "nou ja", "weet je"
- Filler Pauses: "uh", "uuh", "eh", "ehm", "mm", "hm"
- Back-channels: "mm-hmm", "ja", "hmm"

**Speaker Events:**
- (laughs), (sighs), (breathes), (coughs), (clears throat), (claps)

**Other Annotations:**
- Emotion: Happy, Neutral, Sad, Angry, Fearful, Disgusted, Surprised
- Speaker: Male, Female
- Transcription corrections
- Audio quality assessment

## Research Papers

Key research papers and resources informing this project:

* **Parakeet Research**: Comprehensive analysis of speech synthesis approaches - [jordandarefsky.com](https://jordandarefsky.com/blog/2024/parakeet/)
* **Better Speech Synthesis Through Scaling**: Explores scaling laws for speech synthesis models - [arXiv:2305.07243](https://arxiv.org/pdf/2305.07243)
* **Crisper Whisper**: Improved ASR accuracy and forced alignment techniques - [arXiv:2408.16589](https://arxiv.org/pdf/2408.16589) and [PyTorch Forced Alignment Tutorial](https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html)
* **FunAudioLLM**: Foundation models for voice understanding and generation - [funaudiollm.github.io](https://funaudiollm.github.io/)

## TODO

- [ ] Add fluent prompts on transcripts with disfluencies for automatic learning
- [ ] Implement prompt fuzzing using GPT for script variations
- [ ] Improve Speech Emotion Recognition (SER) for Dutch audio
- [ ] Integrate PodcastFillers dataset
- [ ] Mix speaker diversity across different podcast sources
