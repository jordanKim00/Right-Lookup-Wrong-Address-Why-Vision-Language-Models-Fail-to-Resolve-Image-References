# S729

Category: `10_r3_rebinding`. R3 (Read, Resolve, Rebind): calibration, Read selection, the two writes, their mechanism tests and other families.

From the main script's docstring:

> S729: re-calibrate the end reader (READ's count part) over image resolutions -- block-position labels only, no answer, no CIFAR. The S609 / S709 readers were fitted on procedural images only; on blurred low-resolution tiles they misread the last block at N = 9 / 10 (§756: 9B reads block 10 as 9, Qwen3-VL-4B reads block 9 as 10). Same layer and site as the model's current reader, images-first scenes, N 2-10, three image kinds in equal shares: procedural (224 px), natural (caltech256 / food101 rows unused by R³-Ref, 448 px) and the same natural pool degraded (downscaled to 32 / 48 / 64 px, upsca

Code: `code/s729_analyze.py`, `code/s729_reader.py`

Shared modules: `lib/r3_proc.py`, `lib/r3ref.py`, `lib/s588c_reader.py`, `lib/uni.py`

Data: `data/s588p_syneval_fsnew_35_9b.json`, `data/s588p_syneval_fsold_35_9b.json`, `data/s588p_syneval_fsv1_35_9b.json`, `data/s729_analysis.json`, `data/s729_rd_27b_reader.json`, `data/s729_rd_35_4b_reader.json`, `data/s729_rd_35_9b_reader.json`, `data/s729_rd_4b_reader.json`, `data/s729_rd_8b_reader.json`, `data/s729_rd_ivl3_8b_reader.json`, `data/s729_rd_llov_7b_reader.json`, `data/s729_rd_q25_7b_reader.json`
