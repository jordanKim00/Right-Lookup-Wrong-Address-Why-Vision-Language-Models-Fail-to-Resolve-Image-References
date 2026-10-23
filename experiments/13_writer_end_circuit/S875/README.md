# S875

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S875 (advisor review 2, H1-H3: the QK-OV-logit bridge; design note research/ces/outputs/writer_circuit_2026-10-20/main/s875_design_note.md) - does the late lookup's attention on picture b, carried through each head's output gate, value path and W_O, write picture b's menu word into the answer logits (H1); do these direct writes predict the logit margin between the chosen and the runner-up picture (H2); is the answer carried by several heads and layers additively (H3)? Natural runs only (no pins, no patches, no gradients). One scene = nine pictures with all nine names as the menu (shuffled, as 

Code: `code/s875_analyze.py`, `code/s875_bridge.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s875_analysis_s875_27b.json`, `data/s875_s875_27b_bridge.json`
