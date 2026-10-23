# S831b

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S831b (replication of S831's N = 9 cell, whose I0 failed at QNOOP agreement .867; same design and rules, new seed 83200, 60 scenes; pre-registered in knowledge/DECISIONS.md before any S831b data). Original S831 header follows. S831 (pre-registered in knowledge/DECISIONS.md before any S831 data): in non-identity cells, does END's late computation move the early-found position one step toward the end-relative TARGET j (S830 §924: in identity cells the answer moved from a forced early position p one step toward k = j), toward the literal ordinal k, or always one step back (p - 1)? The three rules

Code: `code/s831b_anchor2.py`

Shared modules: `lib/c_qpath_b.py`, `lib/pipe_run.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: none of its own (see below)

Its scripts and records are released with the experiments it was run with (the same notebook section):

- `experiments/05_query_patching/S831/code/s831_analyze.py`
- `experiments/05_query_patching/S831b/code/s831b_anchor2.py`

Notebook sections: §927 (the notebook itself is not released).
