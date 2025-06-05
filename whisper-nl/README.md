# whisper-nl

Whisper fine-tuned on a large set of Dutch audio to output:

- [ ] Disfluencies
- [ ] Speech events

## Data

The CGN contains disfluencies and literal transcriptions of more than 900 hours of Dutch audio. 
The other sets are quite "perfect" because those consists of audio books or short samples.
So probably we should stick to the CGN for now.

- [x] Corpus Gesproken Nederlands (900 hours)
- [ ] JASMIN-CGN (115 hours) - Also mostly text that is read by a human

- [x] Common Voice NL (2 hours)
- [ ] Librivox

### Corpus Gesproken Nederlands

Docs: https://taalmaterialen.ivdnt.org/wp-content/uploads/documentatie/cgn_overhetcgn_nl.pdf

`uv run python -m http.server 8000 --bind 0.0.0.0 --directory ../collector/data/CGN_2.0.3/doc_Dutch/`

- All data contains orthographic transcriptions
- 2,5% of the "Kerncorpus" is annotated with prosodic features (stress, pitch, duration, etc.)
- "Kerncorpus" contains fonetic transcriptions
- Laughing is encoded as "ggg", Unknown is encoded as "xxx". We should filter the unknown out.

### Data Extraction

```
Processing files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12780/12780 [25:53<00:00,  8.23it/s]
2025-06-04 20:17:17,476 - INFO - Metadata saved to ../data/training/metadata.json
2025-06-04 20:17:17,476 - INFO - Whisper dataset saved to ../data/training/whisper_dataset.json
2025-06-04 20:17:17,476 - INFO - CSV dataset saved to ../data/training/dataset.csv
2025-06-04 20:17:17,488 - INFO - === Processing Statistics ===
2025-06-04 20:17:17,488 - INFO - Total chunks before filtering: 104355
2025-06-04 20:17:17,488 - INFO - Chunks with inaudible words: 29693
2025-06-04 20:17:17,488 - INFO - Chunks filtered out: 11
2025-06-04 20:17:17,488 - INFO - Final chunks: 104344
2025-06-04 20:17:17,488 - INFO - Retention rate: 100.0%
2025-06-04 20:17:17,488 - INFO - Total words processed: 8960832
2025-06-04 20:17:17,488 - INFO - Inaudible words removed: 53398
2025-06-04 20:17:17,488 - INFO - Laughter words converted: 66019
2025-06-04 20:17:17,488 - INFO - Inaudible word ratio: 0.6%
2025-06-04 20:17:17,488 - INFO - Laughter word ratio: 0.7%
2025-06-04 20:17:17,564 - INFO - Average inaudible ratio in kept chunks: 0.6%
2025-06-04 20:17:17,564 - INFO - Average chunk duration: 27.0s
2025-06-04 20:17:17,564 - INFO - Average words per chunk: 85.4
2025-06-04 20:17:17,564 - INFO - Average laughter instances per chunk: 0.6
2025-06-04 20:17:17,564 - INFO - Processing complete!
2025-06-04 20:17:17,564 - INFO - Total chunks created: 104344
2025-06-04 20:17:17,564 - INFO - Failed files: 0
2025-06-04 20:17:17,583 - INFO - Total audio duration: 783.74 hours
```

## Evaluation

File: `../collector/test/nl_stutter.mp3`

```
Ja, dat is niet zo makkelijk uit te leggen, zeg maar, weet je?
```

whisper-nl-small:

```
ja uh d dat is niet zo makkelijk uh uit te leggen uh zeg maar weet je
```

It even guessed the short stutter 'd' correctly.

File: `../collector/test/nl_laughter.mp3`

whisper-nl-small:

```
ok ja moet je me meer over vertellen (lacht) dat was een ongemakkelijke lach
```

🚀 OH YEAH!

Even the acknowledgements are handled nicely!

```
... vind dat toch altijd 't leukste wat er is mm-hu en helemaal als er als er iets aan 't doen zijn met 't dier waardoor die nog efficinter wordt mm-hu
```