# S869r

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S869R (registered analysis of stored data; knowledge/DECISIONS.md S869R entry, rules fixed before this analysis was computed) - late readout sufficiency: can the head-specific late attention scores predict the image each run selects, and does the head-mean readout lose that information? Data: S862 G natural END runs (N 6-9, every k, 24 scenes); intervention generalisation on S868 and S865 arm runs. Per run, per image i: f_i = [z_{L,row,h}(i) - mean_i' z_{L,row,h}(i')] over L in {51, 55, 59, 63}, rows B1 / B4 / answer, 24 heads (288). R0a: x_hat = the L47 head-mean argmax anchor a; R0b: a - s(k

Code: `code/s869r_readout.py`, `code/s869r_readout_fixnan.py`

Shared modules: none

Data: `data/s869r_analysis_s869r_27b.json`, `data/s869r_analysis_s869r_fixnan_27b.json`
