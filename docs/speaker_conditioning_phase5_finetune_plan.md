# Speaker Conditioning Phase 5 Finetune Plan

## Objective

Phase 5 is a **tiny, safe fine-tune preparation step**.

The immediate goal is not final voice quality. The immediate goal is:

1. verify that a speaker-conditioned config can be built
2. verify that a local speaker-conditioned batch contains `speaker_id`
3. verify that an old base checkpoint can be loaded into a speaker-conditioned model
4. verify that only speaker module weights are newly initialized

## Checkpoint initialization strategy

The current base checkpoint was trained without speaker conditioning.

That means it does **not** contain:

- `speaker_embedding.weight`
- `speaker_to_encoder.weight`
- `speaker_to_encoder.bias`
- `speaker_to_decoder.weight`
- `speaker_to_decoder.bias`

### Required behavior

When loading the old checkpoint into a new speaker-conditioned model:

- all existing base model weights must load
- only missing speaker module keys are allowed
- any other missing keys must fail
- any unexpected checkpoint keys must fail

This is an explicit compatibility rule, not a silent fallback.

## Why this is safe

The base model remains intact.

Only the new speaker-conditioning parameters are initialized from scratch.

That is the correct setup for a tiny fine-tune:

- pretrained semantic/acoustic knowledge stays
- new speaker embeddings learn how to steer identity

## Dataset reality

Current local dataset:

- one approved real speaker
- about 500 samples
- about 1490 seconds

This is enough for a **smoke experiment**, not for claiming robust multi-speaker coverage.

## What the first experiment should prove

The first experiment only needs to show:

- training starts
- batches contain valid `speaker_id`
- loss is finite
- loss decreases over a few steps
- checkpoint loading is correct

It does **not** need to prove:

- final stable voice quality
- broad multi-speaker generalization
- production readiness

## Recommended first run

Use:

- `config_speaker_conditioned.json`
- `voice_registry/speaker_vocab.json`
- `data/parquet/local_speakers.parquet` or its shard directory
- very small batch size
- very small learning rate
- very small step count

## Interpretation

If the smoke path works:

- Phase 6 can run a tiny fine-tune

If it fails:

- the issue is in config / dataset / checkpoint compatibility
- not in realtime inference
