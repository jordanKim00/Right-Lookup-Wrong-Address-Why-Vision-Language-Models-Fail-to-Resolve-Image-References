# S710

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S710: does the R³ gain on 27B travel through the matcher heads' queries? (repair mediation) Pre-registration in knowledge/DECISIONS.md before any S710 data. Readers and R³-min plan exactly as S709 (exp/s709_eval.py, calibration exp/s709_cal.py). Qwen3.5-27B, new procedural scenes from the dev pool (seed 71000), END questions on hard cells (N = 7, k in {3, 4, 5, 6}; N = 5, k in {3, 4}). Per item: NATIVE (unedited) and R3 (R³-min writes) runs grab the MATCHER heads' queries (pre-RoPE, after q_norm) and output gates at the suffix rows ("?" .. answer) of their layers (S705a discovery set, results/

Code: `code/s710_analyze.py`, `code/s710_mediation.py`, `code/s722_analyze.py`, `code/s722_mediation_all.py`

Shared modules: `lib/c_tools.py`, `lib/r3_build.py`, `lib/r3_build_md.py`, `lib/r3_proc.py`, `lib/r3ref.py`, `lib/s588c_reader.py`, `lib/uni.py`

Data: `data/s710_analysis.json`, `data/s710_mq_27b_mediation.json`, `data/s710_s710s_mediation.json`, `data/s722_mqa_27b_mediation.json`
