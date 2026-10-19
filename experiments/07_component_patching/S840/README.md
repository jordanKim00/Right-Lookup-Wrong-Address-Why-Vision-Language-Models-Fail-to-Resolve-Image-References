# S840

Category: `07_component_patching`. Token-mixer and MLP outputs and residual rows replaced band by band (the writer band 46-49, boundary rows, Shapley shares).

From the main script's docstring:

> S840 (pre-registered in knowledge/DECISIONS.md before any S840 data): S837 found no single band component (layers 46-49) whose direct path to the late queries carries the step (largest: DeltaNet 49, rho .17), the band's direct paths together .46, and the reading mode written by MLP 47. (1) Which group carries the step on its direct paths - the mixers or the MLPs, layers 46-47 or 48-49? (2) Does the step need the band's outputs to be processed downstream, and is any single component a bottleneck? Activation patches (exp/c_qpath_f.py 'cpatch'): END's output of a component at the question mark, b

Code: `code/s840_analyze.py`, `code/s840_paths.py`

Shared modules: `lib/c_qpath_f.py`, `lib/r3_proc.py`, `lib/uni.py`

Data: `data/s840_analysis.json`, `data/s840_s840_27b_paths.json`
