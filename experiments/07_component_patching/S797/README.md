# S797

Category: `07_component_patching`. Token-mixer and MLP outputs and residual rows replaced band by band (the writer band 46-49, boundary rows, Shapley shares).

From the main script's docstring:

> S797 (pre-registered in knowledge/DECISIONS.md before any S797 data): which component writes the address? S795 placed the formation of the lookup-readable address in the answer row's stream at layers 33-35 of Qwen3.5-27B (FINDINGS §855): through the index-matching heads' own query map, the start-relative target share rises from chance after layer 32 to .80-.96 after layer 35, while far end-relative references receive a weakened copy of the ordinal. Here the outputs of single components, at the rows from the question mark to the answer, are exchanged between the natural END(k) run and the START

Code: `code/s797_analyze.py`, `code/s797_writer.py`, `code/s809_circuit.py`

Shared modules: `lib/c_qpath.py`, `lib/r3_proc.py`, `lib/r3_select.py`, `lib/uni.py`

Data: `data/s797_analysis.json`, `data/s797_s797_27b_writer.json`, `data/s809_s809_27b_circuit.json`
