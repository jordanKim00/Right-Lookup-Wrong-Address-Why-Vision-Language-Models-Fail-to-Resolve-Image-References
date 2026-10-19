# S781

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S781: the same-scene positive control for S779 / S780. Can late queries formed in a scene point to that scene's own position? R3 is run afresh in the rearranged scenes B (images ta and N exchanged) and B2 (B's image ta replaced by the first extra word's picture), and its queries (and START(ta)'s from the same scene) are given to that scene's END(5); A's R3 queries into B and B2 replicate S779's donor-content arm; A's R3 write plan applied unchanged to B and B2 checks the fixed plan. Pre-registration in knowledge/DECISIONS.md (2026-10-14 15:1x) before any S781 data; analysis exp/s781_analyze.py

Re-measured with the clean readout in: S783p (files tagged with the re-run name are kept here).

Code: `code/s781_analyze.py`, `code/s781_posctl.py`, `code/s781_run.py`

Shared modules: `lib/c_dmtg.py`, `lib/c_qpath.py`, `lib/c_tools.py`, `lib/r3_min_md.py`, `lib/r3_proc.py`, `lib/r3_select.py`, `lib/uni.py`

Data: `data/s781_analysis.json`, `data/s781_analysis_bare.json`, `data/s781_s781_27b7_posctl.json`, `data/s781_s781_27b8_posctl.json`, `data/s781_s783p_27b7_posctl.json`, `data/s781_s783p_27b8_posctl.json`, `data/s781_sm_s781_posctl.json`
