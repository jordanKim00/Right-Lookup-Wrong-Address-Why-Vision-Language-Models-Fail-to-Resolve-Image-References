# S757

Category: `05_query_patching`. Query edits with keys, values and gates fixed: whole-query swaps, address-part pinning, late-query swaps and transfers.

From the main script's docstring:

> S757: when the option order changes an imprecise END(5) answer, does the L35-47 query matching change with it? Pre-registration in knowledge/DECISIONS.md (2026-10-13 09:3x) before any S757 data; analysis exp/s757_analyze.py. Qwen3.5-27B, seven procedural pictures (160 px), the S749 prompt. Per scene: END(5) with four menu orders (M0 as drawn, M1 reversed, M2, M3 random), END(2) and START(3) with M0 and M1. Readouts: the menu-restricted answer, and at L35, 39, 43, 47, 51, 55 (27B) for the rows "?" (before the menu, so it cannot see it), tail1 (newline after the user turn) and the answer positio

Code: `code/s757_analyze.py`, `code/s757_menuqk.py`

Shared modules: `lib/c_tools.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s757_analysis.json`, `data/s757_s757_27b_menuqk.json`, `data/s757_sm_s757_menuqk.json`
