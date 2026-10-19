# S660

Category: `01_behavior`. Natural answers: scale, answer grids, option-embedded references, text lists, other families and reasoning models.

From the main script's docstring:

> S660: the two skills and their composition on the SAME scene (review W2, 2026-10-07). Procedural scenes as S643 (one-token shapes, N 5-7, restricted content answers over the scene's shapes + 2 distractors). Per scene, with k in 1..min(N,5), j = N+1-k != k: idx "Counting from the start, which image is the {k} image from the end? Answer with a number only." (greedy, first integer) ret "What is in the {j} image?" (explicit forward retrieval of j) comp "What is in the {k} image from the end?" (the composition) Reported: P(comp correct | idx correct and ret correct) with its subset size, and the sh

Re-measured with the clean readout in: S784 (files tagged with the re-run name are kept here).

Code: `code/s660_samescene.py`

Shared modules: `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s660_s784ss_35_4b_samescene.json`, `data/s660_s784ss_35_9b_samescene.json`, `data/s660_s784ss_4b_samescene.json`, `data/s660_s784ss_8b_samescene.json`, `data/s660_ss2_4b_samescene.json`, `data/s660_ss_35_4b_samescene.json`, `data/s660_ss_35_9b_samescene.json`, `data/s660_ss_4b_samescene.json`, `data/s660_ss_8b_samescene.json`
