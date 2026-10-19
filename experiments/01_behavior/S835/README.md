# S835

Category: `01_behavior`. Natural answers: scale, answer grids, option-embedded references, text lists, other families and reasoning models.

From the main script's docstring:

> S835 (descriptive re-measure, registered in knowledge/DECISIONS.md before any S835 data): which signature do Qwen3.5-27B's wrong answers on end-relative MMIU options carry, under a clean readout? §928 (exp/p2_w3_mmiu_signature.py) measured it on S489's saved answers (single-form restricted option-letter argmax, no free-generation check): literal .599 vs chance .256, step .142 vs chance .243. Items, prompt and pictures are S489's (exp/s489_mmiu_fix.py, the S462 items): the eleven MMIU tasks whose options all read 'X: The m-th image', at most 100 per task, N <= 9, longer side <= 448 px; the F (f

Code: `code/s835_analyze.py`, `code/s835_mmiusig.py`

Shared modules: `lib/uni.py`

Data: `data/s835_analysis.json`, `data/s835_s835_27b_mmiusig.json`
