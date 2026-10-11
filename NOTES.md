# Retrieval notes

## Day 6: threshold and chunking

Eval set: `eval/questions.txt` (7 answerable, 6 must-refuse). Run: `python -W ignore main.py eval`.
Metric: does "any hit under `--max-distance`" match the expected answer/refuse label.

### Chunk size sweep (threshold 0.45, top-k 3)

| size/overlap | chunks | score |
|---|---|---|
| 40/10 | 29 | 10/13 |
| 80/20 | 17 | 10/13 |
| 120/30 | 14 | 10/13 |

Chunk size changes little: 9 of 11 docs are under 80 words and stay one chunk at any setting.
On the two long docs, 40/10 tightens good hits (0.235 -> 0.204, 0.354 -> 0.281) but also pulls
noise closer (nginx 0.436 -> 0.415). Keeping 80/20.

### Threshold

Same index (80/20): 0.45 -> 10/13, 0.40 -> 12/13. Default changed to 0.40.
Margin is thin: hardest answerable question peaks at 0.365, nearest must-refuse at 0.416.
Re-check after adding documents.

Distance scale (nomic-embed-text, cosine): < 0.30 strong, 0.30-0.40 related, > 0.45 noise.

### What distance cannot catch

"gather vs asyncio.wait" scores 0.252: right topic, fact absent from the base. Only the prompt rule
("answer only from context, else say I don't know") stops this one. Two layers needed:
threshold for off-topic, prompt rule for on-topic-but-missing.
