# S831

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S831 (pre-registered in knowledge/DECISIONS.md before any S831 data): in non-identity cells, does END's late computation move the early-found position one step toward the end-relative TARGET j (S830 §924: in identity cells the answer moved from a forced early position p one step toward k = j), toward the literal ordinal k, or always one step back (p - 1)? The three rules disagree on probes between j and k and below j. Cases (N, k, j): N = 7 END(5) (j = 3), N = 9 END(7) (j = 3). Settings exactly as the main series; seed 83100 (new), 30 scenes per case. Per scene: natural END(k), START(k) and ST

Code: `code/s831_analyze.py`, `code/s831_anchor2.py`

Shared modules: `lib/c_qpath_b.py`, `lib/pipe_run.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s831_analysis.json`, `data/s831_analysis_s831b_27b.json`, `data/s831_analysis_s838nid_35_4b.json`, `data/s831_analysis_s838nid_35_9b.json`, `data/s831_s831_27b_anchor.json`, `data/s831_s831b_27b_anchor.json`, `data/s831_s838nid_35_4b_anchor.json`, `data/s831_s838nid_35_9b_anchor.json`
