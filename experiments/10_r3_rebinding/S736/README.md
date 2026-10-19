# S736

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S736: when is the frame write needed, and is its direction what matters? (pre-registration knowledge/DECISIONS.md 2026-10-10 16:35) The S730 harness (same Read, writes, matcher heads, readouts) over scene sizes N in {5, 7, 9} (20 scenes each, seed 73600 + N) and every k = 1..N except the middle; written address t = N + 1 - k. Arms per item: START forward wording of the same target END end-relative wording, unedited ADDR R³'s ordinal (address) write only R3 both writes RANDF the address write + a random direction of the same norm at each frame-write site and layer (seed 73600 + 1000 scene + 10 

Re-measured with the clean readout in: S787 (files tagged with the re-run name are kept here).

Code: `code/s736_analyze.py`, `code/s736_frame.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/r3_select.py`, `lib/uni.py`

Data: `data/s736_analysis.json`, `data/s736_fr736_27b_frame.json`, `data/s736_fr736_35_4b_frame.json`, `data/s736_fr736_35_9b_frame.json`, `data/s736_fr736_4b_frame.json`, `data/s736_fr736_8b_frame.json`, `data/s736_s787_27b_frame.json`, `data/s736_s787_35_4b_frame.json`, `data/s736_s787_35_9b_frame.json`, `data/s736_s787_4b_frame.json`, `data/s736_s787_8b_frame.json`
