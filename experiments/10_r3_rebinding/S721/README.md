# S721

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S721 latency / memory: per-input wall time of the unedited forward vs R³ (capture forward + plan + repaired forward when something is written) on MMIU development items (end-relative and forward forms), and peak GPU memory. Usage: python s721_latency.py MODEL TAG --base B [--per_task 10] [R³-v3 flags] [--device_map balanced]

Code: `code/s721_analyze.py`, `code/s721_latency.py`

Shared modules: `lib/r3_min_md.py`, `lib/r3_select.py`, `lib/uni.py`

Data: `data/s721_analysis.json`, `data/s721_lat_27b_latency.json`, `data/s721_lat_4b_latency.json`
