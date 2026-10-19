# S849u

Category: `07_component_patching`. Token-mixer and MLP outputs and residual rows replaced band by band (the writer band 46-49, boundary rows, Shapley shares).

From the main script's docstring:

> S849-U analysis (rules fixed in knowledge/DECISIONS.md before any S849 data). Test A part 2: is the goal-attainment signal found by S848 used to stop the update? 8 pictures, k = 4, N_s 6-9, early lookup pinned to p 3-6 (match cells: p = N_s - 3); the answer row's residual steered at the input of layer 46 or 50 along the frozen match direction m (+ALPHA: PLUS, -ALPHA: MINUS) or a random direction r of the same norm in the same 50-PC subspace (RAND). stay = P(answer = p). Paired scene bootstrap (4,000, seed 0). I0 (per N_s; all must pass): t1 >= .95; QNOOP_E = END in >= .90 of scenes, 95th perce

Code: `code/s849u_analyze.py`

Shared modules: none

Data: `data/s849u_analysis.json`, `data/s849u_analysis_s849u2_27b.json`
