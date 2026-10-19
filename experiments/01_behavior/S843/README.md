# S843

Category: `01_behavior`. Natural answers: scale, answer grids, option-embedded references, text lists, other families and reasoning models.

From the main script's docstring:

> S843 (pre-registered in knowledge/DECISIONS.md before any S843 data): text modality, user order (10-17: after the gaps) - do word lists show the same end-relative pattern as pictures? The image main series asks for the k-th of N pictures; here the items are the label words themselves, in one line ('Words: w1, w2, ...'), with the same sentence otherwise: 'There are N words in a row. What is the k-th word from the last / first? Answer with only one word from: <menu>.' Label pool, menu recipe (the N words + the first unused words to nine, shuffled), THINK prefix (Qwen3.5), two-form menu readout a

Code: `code/s843_analyze.py`, `code/s843_textgrid.py`

Shared modules: `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s843_analysis_s843_27b.json`, `data/s843_s843_27b_textgrid.json`
