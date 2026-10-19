# S771

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S771: the sender -> reader connection. How does the message that sets the answer reach the full-attention queries at L43-59 (the readers that control the answer, S767), and where does END(5)'s error enter it? Query-path patching (exp/c_qpath.py). Pre-registration in knowledge/DECISIONS.md (2026-10-14 00:2x) before any S771 data; analysis exp/s771_analyze.py. Qwen3.5-27B, seven procedural pictures (160 px), the S749 prompt, new scenes (seed 77100). Per scene END(5), START(5), END(2), START(2); F5P1 = END(5) whose rows after the frame word read START(5)'s frame word at L27-34 only (S758 read int

Code: `code/s771_analyze.py`, `code/s771_qpath.py`, `code/s773_analyze.py`, `code/s773_writers.py`

Shared modules: `lib/c_qpath.py`, `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s771_analysis.json`, `data/s771_s771_27b_qpath.json`, `data/s771_sm_s771_qpath.json`, `data/s771_sm_s771b_qpath.json`, `data/s771_sm_s771c_qpath.json`, `data/s773_s773_27b_writers.json`
