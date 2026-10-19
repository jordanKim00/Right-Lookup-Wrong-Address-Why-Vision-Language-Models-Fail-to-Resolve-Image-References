# Data

## What is released

| Folder | What | Format |
|---|---|---|
| `experiments/*/S*/data/` | registered analyses (`*_analysis*.json`) and per-run records of each experiment | JSON |
| `calibration/` | R3 calibration (ordinal and frame means, chart projection, detector thresholds, layers) and index readers | JSON, tensors as `{"shape", "dtype", "data"}`; floats to 9 significant digits, which round-trip float32 |
| `paper_builders/inputs/` | small derived files read by the figure and table builders (answer grids, frozen laws, query-key tables, benchmark values) | JSON, TeX |
| `manifests/files.json` | every released file: original path in the workspace, released path, size, SHA-256 of the original | JSON |
| `manifests/omitted.json` | activation captures that are not released: original path, size, SHA-256 | JSON |

## Per-run records

A run record is a JSON object with the run's settings (model, tag, layers, heads, seed) and `rows`, one per scene or item.

- **Answers.** Each row holds the answer under each arm (`land`). Positions are 1-based image indices, and 0 means a menu word not in the scene.
- **Logits.** `mlog` holds the menu-word logits used by the analyses.
- **Other per-arm values** depend on the experiment and are named in its script's docstring.

## Slimmed files

Records larger than 20 MB are released with their heavy per-row arrays removed. These are typically `qk` (block scores per head) and
`mass` (image-attention mass). The answers and logits are kept. The removed keys are listed per file in `manifests/files.json` (`slimmed`,
`dropped_keys`), with the SHA-256 of the full original. The registered analyses (`*_analysis*.json`) are never slimmed.

## Not released

- **Activation captures.** The `.npz` files hold query–key logits, block scores and residual states (about 9 GB). They are listed in
  `manifests/omitted.json`. The analyses that read them saved their outputs as JSON, and those outputs are released.
- **Model weights.** These are the public checkpoints listed in `docs/REPRODUCE.md`.
- **Benchmark images and texts.** MMIU and MuirBench are not redistributed. The released per-item results hold the task, gold letter, chosen
  letters and the R3 read plan.
- **Photographs.** CIFAR-10 photographs used by the index-code experiments are loaded from the dataset itself.
- **Smoke tests.** CPU and smoke runs are software checks, not evidence, and are not released.

## Readout tags

Files of re-measurements carry the re-run's tag, for example:

| File | Re-run |
|---|---|
| `s545_indexshift_s787_27b.json` | S787 re-run of S545 |
| `s769_analysis_bare.json` | S783a re-analysis with the corrected readout |
| `s730_s786_27b_stagewise.json` | S786 |

Only clean-readout values are used in the manuscript. The originals are kept for provenance.
