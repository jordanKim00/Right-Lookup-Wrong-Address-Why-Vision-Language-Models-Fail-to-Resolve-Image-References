# S616b

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S616b: choose the write layer inside the model by forward rebinding efficacy, then rebuild S609c and evaluate (pre-registered, exp/QUEUE.md S616b). One model load. Efficacy at layer L (forward items only, procedural pictures): question "What is in the k-th image?" -- write Mb_L[k'] - Mb_L[k] on the ordinal, success = the answer names block k'; options "Which image shows a <label of block j>?" whose options do not point to j -- rewrite option X's ordinal to j, success = option X is chosen. Write layer = argmax over L of min(question efficacy, options efficacy), ties -> the earlier layer. Usage:

Code: `code/s616b_pipe.py`

Shared modules: `lib/r3_build.py`, `lib/r3_proc.py`, `lib/r3_repair.py`, `lib/r3ref.py`, `lib/s608_layout.py`, `lib/uni.py`

Data: `data/s616b_ivl3_8b_writelayer.json`, `data/s616b_llov_7b_writelayer.json`, `data/s616b_q25_7b_writelayer.json`, `data/s616b_sw_35_4b_writelayer.json`, `data/s616b_sw_35_9b_writelayer.json`, `data/s616b_sw_4b_writelayer.json`, `data/s616b_sw_8b_writelayer.json`
