# S782

Category: `00_readout_audit`. Validity of the answer readout (S782) and the clean-readout re-measurements of every affected experiment (S783-S794, S789).

From the main script's docstring:

> S782: measurement validity - does the menu-restricted argmax at the answer position agree with what the model generates freely? Same scenes as S780 (seeds 78000 / 78001, N = 7 / 8, wording "last"). Per scene END(5), START(N - 4), END(2) and R3 on END(5) (exp/r3_select.build: model-specific READ, write layer L3): greedy generation (8 new tokens) with the first step's scores; restricted answer = argmax of the first step over the menu words' first tokens; free answer = the first word of the generated text (lowercased, punctuation stripped). Pre-registration in knowledge/DECISIONS.md before any S7

Code: `code/s782_analyze.py`, `code/s782_freegen.py`, `code/s782_run.py`

Shared modules: `lib/r3_proc.py`, `lib/r3_select.py`, `lib/uni.py`

Data: `data/p2_w3_mmiu_signature.json`, `data/s782_analysis.json`, `data/s782_s782_27b7_freegen.json`, `data/s782_s782_27b8_freegen.json`
