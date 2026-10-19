# R3 calibration

All components of R3 are fitted on items of known structure. No end-relative answer and no benchmark image is used. The tensors are
exported from the original `.pt` files. Each tensor is stored as `{"shape", "dtype", "data"}`, with floats written to 9 significant digits.

| File | Contents |
|---|---|
| `s709_<base>_cal.json` | `cal` and the layers. `cal` holds the ordinal means `Mb` and `Mr` (10 × d), the chart projection `Pbb` (d × 9), the detector thresholds `th_chart` and `th_full` with the space they apply in, and the frame means `mu_start`, `mu[end]`, `mu[last]` with their radii `thr`. The layers are `Lb` (read) and `Lw` (write). Also stored: the index reader's name and its accuracy on calibration sequences per layer and site. |
| `s588c_reader_<name>.json` | Index readers. Each holds a standardization (`m`, `s`), principal axes and ten-class logistic weights over the states of chosen sites (block mean, markers) at chosen layers. |
| `read_plan_s738.json` | The Read chosen for each model from the fixed grid: detector rule, frame binder and count rule. It also gives the grid itself and the exact-plan rate on the confirmation split. |

Base tags and models:

| Tag | Model |
|---|---|
| `lm_4b` | Qwen3-VL-4B |
| `lm_8b` | Qwen3-VL-8B |
| `lm_35_4b` | Qwen3.5-4B |
| `lm_35_9b` | Qwen3.5-9B |
| `r327b` | Qwen3.5-27B |
| `ivl3_8b` | InternVL3-8B |
| `llov_7b` | LLaVA-OneVision-7B |
| `q25_7b` | Qwen2.5-VL-7B |

Suffixes `w1`, `w7` or a trailing `4` mark calibrations made at another write layer, used in the write-layer comparison. Readers `s729_*`
are the recalibrated count readers used by the final R3. The item-level calibration records used to choose the Read are in
`experiments/10_r3_rebinding/S724/data/`.

R3 code: `lib/r3_min_md.py` (the two-pass repair), `lib/r3_build.py` (clause windows, calibration helpers), `lib/s588c_reader.py`
(index reader) and `lib/r3_select.py` (the per-model Read).
