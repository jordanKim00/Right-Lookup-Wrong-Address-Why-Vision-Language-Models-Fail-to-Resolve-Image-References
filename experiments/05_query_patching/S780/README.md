# S780

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S780 (validation of S779 on new scenes, N = 7 and N = 8; adds the L35-39 query arm and the visual source of near END(2)). S779: the address-to-answer handoff. Does the repairing query route the lookup to a physical image position (address) or carry the donor's answer (content), does the paper's repair R3 work through the same queries and parts, and do natural END errors and R3 answers read the image they name? Pre-registration in knowledge/DECISIONS.md (2026-10-14 15:0x) before any S780 data; analysis exp/s780_analyze.py. Qwen3.5-27B, procedural pictures (160 px), the S749 prompt with the stat

Re-measured with the clean readout in: S783a (files tagged with the re-run name are kept here).

Code: `code/s780_analyze.py`, `code/s780_handoff.py`, `code/s780_run.py`

Shared modules: `lib/c_dmtg.py`, `lib/c_qpath.py`, `lib/c_tools.py`, `lib/r3_min_md.py`, `lib/r3_proc.py`, `lib/r3_select.py`, `lib/uni.py`

Data: `data/s780_analysis.json`, `data/s780_analysis_bare.json`, `data/s780_s780_27b7_handoff.json`, `data/s780_s780_27b8_handoff.json`, `data/s780_s783_27b7_handoff.json`, `data/s780_s783_27b8_handoff.json`, `data/s780_sm_s780_handoff.json`
