# Right Lookup, Wrong Address: Why Vision–Language Models Fail to Resolve Image References

Code and data for a mechanistic study of end-relative image references ("What is in the third image from the end?") in multi-image
vision–language models (Qwen3-VL-4B/8B and Qwen3.5-4B/9B/27B).

The study separates two parts of a reference to an image:

- **The lookup.** Images are retrieved by matching an address in the query against an index code that every image block carries, counted from
  the first image. Any address written into the query selects the image at that address.
- **The address.** The end frame is not resolved into the absolute position `N + 1 - k`.
  - Smaller models copy the ordinal.
  - Qwen3.5-27B also copies it first, then applies an end-conditioned late update. Layers 46–49 write the update and the last full-attention
    queries (layers 51–63) read it.
  - The update is not stopped when the copy is already right, so the identity reference (the middle image asked from the end) breaks.
  - The images actually shown reach the same late queries through full-attention reads in layers 35–50.
- **The repair.** R3 (Read, Resolve, Rebind) reads the ordinal, its frame and the image count from the model's states, computes `N + 1 - k`, and
  rewrites the ordinal and the frame in a second pass through the frozen model.

All experiments are interventions on hidden states with predictions registered before their data. This repository contains the code of
every experiment used in the manuscript versions and the result tables needed to check each number, as JSON.

## Layout

```
README.md
docs/
  EXPERIMENTS.md      every experiment: category, code, data, which manuscript version uses it
  INTERVENTIONS.md    how each method touches layers and hidden states (hooks, rows, routes), with pointers to the code
  DATA.md             data formats, slimmed files, what is not released
  REPRODUCE.md        container, checkpoints, prompts, seeds, running a script
lib/                  shared modules (model wrapper, attention and DeltaNet editing tools, R3, readers, procedural scenes)
experiments/
  00_readout_audit/          answer-readout validity and the clean-readout re-measurements
  01_behavior/               natural answers, scale, answer grids, option-embedded references, text lists, other families
  02_qk_lookup/              query-key captures, index-matching heads, address lens, nearest start-relative lookup
  03_index_code_and_copy/    index-code shifts, key/value exchange, block size, copy-versus-compose tests
  04_activation_patching/    residual-stream transplants and writes at the ordinal token
  05_query_patching/         query edits with keys, values and gates fixed (whole query, address part, late queries)
  06_read_interchange/       route-complete read interchanges through full attention, the DeltaNet state and its convolution
  07_component_patching/     token-mixer / MLP outputs and residual rows (the writer band 46–49, boundary rows, Shapley)
  08_deltanet_source_cut/    full-attention key masks and the exact segmented Gated DeltaNet cut
  09_count_pathway/          image count shown versus stated, rotary geometry
  10_r3_rebinding/           R3: calibration, Read selection, the two writes and their mechanism tests
  11_benchmarks/             MMIU / MuirBench harness and per-item results
  12_laws_and_predictions/   answer laws and predictions fixed in advance, held-out source-cut predictions
  13_writer_end_circuit/     the END circuit of 27B: frame read, early lookup, the END-conditioned rewrite (46–49), late reads, R3 against it
  14_intervention_response_prediction/  fitted profile predictors and the parameter-free intervention-response prediction
  15_text_only/              the same instruments on text-only word lists
    S<ID>/
      README.md       what the experiment asks, its code, data and shared modules
      code/           exact copies of the scripts that ran
      data/           result tables as JSON (registered analyses and per-scene records)
analysis/posthoc/     post hoc analyses (labelled as such in the manuscript); writer_circuit/ holds those of S913–S932 and the frozen predictors
calibration/          R3 calibration tensors and index readers, exported to JSON
paper_builders/       figure, table and number-check scripts of every manuscript version, with the small files they read
manifests/            files.json (original path, released path, SHA-256), omitted.json (activation captures not released)
tools/link_layout.py  recreates the original workspace layout as symbolic links so the scripts run unchanged
```

Experiment IDs (S545, S857, …) follow the order in which the experiments were registered. `docs/EXPERIMENTS.md` maps every ID to its
category and to the manuscript versions that use it.

## Where the main results come from

| Result | Experiments |
|---|---|
| The lookup reads an index code counted over the images | S545 (re-run in S787), S741b and S739 (re-run in S789), S705a |
| A written address selects its image | address-control audit (`10_r3_rebinding`), S708 (re-run in S789) |
| Smaller models copy the ordinal | S801, S673b (re-run in S789), S720 (re-run in S789), S787 |
| The frame word is read at layers 27–34 and acts through the lookup queries | S769 (re-run in S783a), S796, S758/S759 |
| The early lookup copies, then a late update moves the address and is not stopped at the target | S795, S827–S834, S844, S846, S847 |
| Layers 46–49 write the update, layers 51–63 read it | S845–S853, S854-2 |
| The end comes from the images shown, through full attention in layers 35–50 | S748, S854-1, S855, S856b, S857 |
| Predictions fixed in advance hold in held-out input configurations | S858 |
| Both writes are needed, and the rebound reference reads the picture at its address | S786/S794, S787, S791, S793, S783a, S740, S833 |
| Benchmarks | S582 harness, final runs `f7*` (four smaller models) and `c8*` (27B) |
| The END state is read into the read rows at layers 27–34 and re-enters them from the frame-word and menu rows | S896, S898, S903, S906, S909, S916 |
| The early lookup (35–47) forms a graded preference over the images; swapping it between END runs moves the answer as predicted | S913, S915 |
| The END-conditioned rewrite in layers 46–49 (mostly the MLP of layer 46) moves the address, also when it was already right | S859–S887, S912, S915, S918, S919 |
| In output swaps the rewrite moves with the late full-attention read outputs (55–63), more with the menu-name reads than the image reads | S917, S921 |
| The first-order intervention response is carried within the read rows (97%); the late full-attention reads alone carry 55–60%, the late DeltaNet mixers and MLPs the rest | S929 |
| R3 against the circuit: address write alone = the natural END(target) question, frame write alone = START(k), both = START(target) | S920, S922, S923 |
| Parameter-free intervention-response prediction on fresh scenes, including intervention types never run before | S926, S927, S930 (S928, S932 development) |
| Text-only word lists: analogous END-conditioned distortions, captured by the same intervention-response framework | S931 |

## Methods in one paragraph each

Details, hooks and code pointers are in `docs/INTERVENTIONS.md`.

- **Block scores.** For a head, the log-sum-exp of its scaled query–key logits over the tokens of each image block, normalized across blocks.
  Images precede the question, so image keys are identical in start- and end-relative questions of a scene. Every change of the scores is a
  change of the query.
- **Query edits** (`lib/c_tools.py`, `lib/c_pat.py`, `lib/c_qpath*.py`). At chosen rows and full-attention layers the attention output is
  recomputed in float32 with the query, or its block-mean address part, taken from a donor run. Keys, values and output gates stay fixed.
- **Residual transplants and writes.** A forward hook on a decoder layer replaces or shifts the residual state of chosen tokens, either
  from a donor run or by adding calibrated mean differences (R3).
- **Read interchange.** In a band of layers, every later token reads one source token as it appears in a donor question. This runs through
  every route: full-attention keys and values, the DeltaNet memory entries (q, k, v, g, β) and the four-tap convolution windows. The source
  row's own output is restored.
- **Component patching.** The outputs of chosen token mixers and MLPs, or whole residual rows, are replaced by those of another run of the
  same scene in the same forward pass.
- **Source cut** (`lib/c_cut.py`). The last images are removed from the text rows through every route of the hybrid model:
  - full attention gets explicit boolean masks;
  - the chunked gated delta rule runs in three segments, so the rows after the images start from the state before them;
  - the convolution windows are spliced across the boundary.
  A finite decay is not used, because it leaks through the in-chunk cumulative sum.

- **Intervention-response prediction** (`experiments/14_intervention_response_prediction/`). For a natural run, the change of the
  menu logits under an intervention is predicted from the model's own gradients at the circuit's sites times the intervention's exact local
  change there (the pinned attention recomputed from the natural inputs; the replaced band outputs), with the gradients taken with replaced
  outputs held fixed. Band replacement crosses a curved region, so its part is integrated along its own path with one more evaluation at the
  half-replaced band (a quadratic Hermite rule). Nothing is fitted to any outcome; the method uses the model's gradients and partial
  interventions, never the full intervention being predicted. The transfer matrices it produces (pin the address to q, read the change of
  every answer) show the k-specific step of the END circuit directly.

## Readout

Procedural answers are the menu word with the highest next-token logit in either tokenization (with or without a leading space), set by
the environment variable `R3_READ` (`max`; `bare` for the space-free form). An earlier readout restricted to the space-prefixed form
disagreed with free generation on uncertain answers (S782).

- Every answer-based result affected by that readout was re-measured: S783–S794, S787, S789.
- Re-measured files carry the re-run's tag and sit next to the original experiment.
- Benchmark answers are option letters and were not affected.

## Data

- **Format.** JSON only.
- **Large files.** Files above 20 MB are released with their heavy per-row arrays removed. The removed keys and the original SHA-256 are
  listed in `manifests/files.json`.
- **Not released.** Activation captures (`.npz`) are not released. `manifests/omitted.json` lists them with size and SHA-256.
- **Calibration.** R3 calibration and the index readers are in `calibration/`. Each tensor is stored as `{"shape", "dtype", "data"}`.

See `docs/DATA.md`.

## Reproducing

See `docs/REPRODUCE.md`.

- **Software.** One container with Python 3.11.10, PyTorch 2.5.1 (CUDA 12.1) and Transformers 5.16.1.
- **Checkpoints.** The public Qwen checkpoints.
- **Run layout.** Every script was run with the workspace mounted at `/vlm`. Run `python tools/link_layout.py` once to recreate that layout
  (`exp/`, `results/`, `research/`) inside this repository, mount the repository at `/vlm`, and the scripts run unchanged.
