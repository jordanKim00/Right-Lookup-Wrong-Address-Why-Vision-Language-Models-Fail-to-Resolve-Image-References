# S723

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S723: the S710 repair-mediation test in a smaller model (Qwen3.5-9B) -- does the repair also travel through the index-matching heads' queries when the original failure is a literal query? Pre-registration in knowledge/DECISIONS.md before any S723 data. Latest R³ plan (R³-v3 rules, as exp/r3_min_md.py) on the earlier module's calibration for Qwen3.5-9B (base lm_35_9b: write layer 4, reader s609_35_9b); index-matching heads = exp/s720_heads.py set (16 heads at L23/27). New dev scenes (seed 72300), N7 k3-6 and N5 k3-4, 25 scenes per N, positions from "?" to the answer. Arms as S710: R3_QBACK / R3

Code: `code/s723_analyze.py`, `code/s723_mediation_small.py`

Shared modules: `lib/c_tools.py`, `lib/r3_build.py`, `lib/r3_build_md.py`, `lib/r3_proc.py`, `lib/r3ref.py`, `lib/s588c_reader.py`, `lib/uni.py`

Data: `data/s723_analysis.json`, `data/s723_mq_9b_mediation.json`, `data/s723_s723s_mediation.json`, `data/s738c_pc738_35_4b_plancheck.json`, `data/s738c_pc738_35_9b_plancheck.json`, `data/s738c_pc738_4b_plancheck.json`, `data/s738c_pc738_8b_plancheck.json`
