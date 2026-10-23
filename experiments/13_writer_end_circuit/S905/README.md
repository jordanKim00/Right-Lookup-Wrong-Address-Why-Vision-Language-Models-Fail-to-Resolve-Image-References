# S905

Category: `13_writer_end_circuit`. The END circuit of Qwen3.5-27B: writer-band anatomy (46-49), the frame read, the early lookup, the END-conditioned rewrite, the late full-attention reads, and R3 against the circuit on the same scenes.

From the main script's docstring:

> S905 (registered in knowledge/DECISIONS.md before its data) - does the writer band move the address toward the end-relative target within the end route's reach and fall back to a fixed one-step-back beyond it? S902: at the natural last-picture anchor END(1)'s band holds while END(2)'s steps back; S863: END(2) pinned at 4-5 does not step back; S872 scenes: END(5) steps back at pins 3-5; S892 / S899 post hoc: natural anchors before a near-end target move forward, on it hold, after it step back, while the middle identity cell (N9 END(5)) leaves the target. Grid at N 8: END(k) (target t = 9 - k) p

Code: `code/s905_analyze.py`, `code/s905_nearend.py`

Shared modules: `lib/c_headpatch.py`, `lib/c_qpath.py`, `lib/c_qpath_k.py`, `lib/c_tools.py`, `lib/c_tritonlock.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s905_analysis_s905_27b.json`, `data/s905_s905_27b_nearend.json`
