# S779

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S779: the address-to-answer handoff. Does the repairing query route the lookup to a physical image position (address) or carry the donor's answer (content), does the paper's repair R3 work through the same queries and parts, and do natural END errors and R3 answers read the image they name? Pre-registration in knowledge/DECISIONS.md (2026-10-14 14:3x) before any S779 data; analysis exp/s779_analyze.py. Qwen3.5-27B, procedural pictures (160 px), the S749 prompt with the stated count, wording "last". E5 = END(5) (target ta = N - 4), S = START(ta). Scenes per draw: A (original); B = A with images

Code: `code/s779_analyze.py`, `code/s779_handoff.py`

Shared modules: `lib/c_dmtg.py`, `lib/c_qpath.py`, `lib/c_tools.py`, `lib/r3_min_md.py`, `lib/r3_proc.py`, `lib/r3_select.py`, `lib/uni.py`

Data: `data/s779_analysis.json`, `data/s779_s779_27b7_handoff.json`, `data/s779_sm_s779_handoff.json`
