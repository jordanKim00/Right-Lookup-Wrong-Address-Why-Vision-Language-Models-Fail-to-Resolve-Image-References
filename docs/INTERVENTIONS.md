# How the experiments touch the model

All edits act on the frozen model at inference. The model wrapper is `lib/uni.py` (`Uni`). It loads a checkpoint in bfloat16 with SDPA
attention, builds the chat prompt, exposes the decoder layers as `v.layers`, and reads the menu-word logits.

Qwen3.5 is a hybrid:

- Gated DeltaNet layers, with full attention at every fourth layer: layers 3, 7, …, 63 in the 64-layer 27B model.
- Full attention uses grouped-query heads with an output gate and a partial multimodal rotary code.
- Each Gated DeltaNet layer has a four-tap causal convolution on q/k/v, a write strength β, a decay g and a matrix-valued state.

Rows, layers and heads are always chosen per experiment and recorded in its script's docstring and arguments.

## Rows and readouts

- **Rows.** The text rows that are read or edited are the question mark, the five chat-boundary rows that follow the question (B0–B4) and
  the answer rows. The experiments call these the read rows.
- **Block score.** For a head and a row, the block score of image b is `logsumexp_{i in b} q·K_i / sqrt(d)` over the image's tokens.
  Block shares are its softmax over blocks. Capture code: `experiments/02_qk_lookup/S705a/code/s705a_qkanatomy.py`.
- **Index-matching heads.** Full-attention heads that give the referenced block most of their image attention for start-relative questions:
  73 heads in layers 35–55 of 27B.
- **Address lens** (`experiments/02_qk_lookup/S795/code/s795_lens.py`). It passes the answer row's residual after layer l through a later
  index-matching layer's input norm, query projection, query norm and rotary code, and scores the result against that layer's own image keys.

## Query edits (keys, values and gates fixed)

- **`lib/c_tools.py`, `FullAttn` / `Ctx`.** A forward hook on a full-attention layer recomputes the output rows of chosen query positions in
  float32 with components exchanged:
  - the query (pre-RoPE, after the query norm; whole, rotary part or pass-through part)
  - the output gate
  - keys or key positions between position sets
  - values
  A no-op recompute is the control for the recomputation itself.
- **`lib/c_tools.py`, `DeltaNet`.** It recomputes a Gated DeltaNet layer's output rows exactly,
  `o_t = Σ_s w(t,s) v_s`, `w(t,s) = β_s k_s · A_{s+1}…A_t q_t`, `A_j = exp(g_j)(I − β_j k_j k_jᵀ)`, with the row queries or the memory
  (k, v, β, g) taken from a donor run.
- **`lib/c_pat.py` (`PatAttn`) and `lib/c_qpath*.py`.** Attention-pattern edits and query-path tools used by the later experiments.
  - The block-mean address part of a query is its projection `P_U q` onto `U = span(centred block-mean image keys)`.
  - An address edit sets `q ← q − P_U q + P_U q_donor`. The non-address edit keeps `P_U q` and takes the complement from the donor.
  - Examples: S827–S834 (pinning the early lookup at layers 35–47), S844, S846.
- **`lib/c_cut.py`, `QFrom`.** It swaps the late queries within one forward pass: at chosen full-attention layers the pre-RoPE query of a
  sample's read rows is replaced by another sample's query, computed in the same pass from that sample's own hidden states. With the sample
  as its own donor, the run reproduces itself bit for bit (S856b).

## Residual transplants and writes

- **Mechanics.** A forward hook on decoder layer `l` replaces, for chosen token positions, the residual output with a donor's state, or adds
  a vector to it.
- **Index-code shift** (`experiments/03_index_code_and_copy/S545/code/s545_indexshift.py`).
  - The donor scene shows two new images before the recipient's.
  - At every layer, the states of every token of each image block are replaced by those of the donor block with the same picture.
  - Content and order stay the same and only each block's code rises by two.
- **Key/value exchange** (`S741b`). It exchanges only the keys and values of the image tokens in every full-attention layer.
- **Ordinal state as a position** (`S673b`) **and the forward-ordinal transplant** (`S474b`). The ordinal token's residual at one layer is
  replaced by that of a start-relative donor question.
- **R3 writes** (`lib/r3_min_md.py`).
  - At layer `ℓ_w` (1 in Qwen3-VL-4B, 3 in Qwen3.5-27B, 4 elsewhere), during prefill, the hook adds `μ_ord[ĵ] − μ_ord[k̂]` to the ordinal
    token and `μ_frame[START] − μ_frame[a]` to the frame word.
  - The means are calibrated and stored in `calibration/`.
- **Targeted writes** (`S708`, `S791`). The same hook writes a chosen address, correct or wrong.

## Read interchange (route-complete)

`experiments/06_read_interchange/S758/code/s758_paths.py` (and S759, S769, S770): a source token p is chosen, an ordinal or the frame word.

- **Recipient input.** At every layer of a band B, the layer's sequence mixer receives the donor question's normalized input at position p.
- **Effect.** Every later row therefore reads the donor's p through full-attention keys and values, the DeltaNet memory entries and the
  convolution windows of positions p+1…p+3.
- **Source row.** p's own output is restored to the recipient's, so only what later rows read from p changes.
- **Single-route arms.** Variants exchange only the convolution or only the memory.

This identified the reads of the ordinal (layers 0–22 plus the layer-26 convolution) and of the frame word (layers 27–34).

## Component and row patching

Hooks on the token mixer and the MLP of chosen layers replace their outputs at the read rows with those of another run of the same scene,
in the same forward pass.

| Experiment | What it patches |
|---|---|
| S845–S847 | The writer band, layers 46–49, between start- and end-relative questions pinned to the same early position |
| S853 | Residual rows B1 and B4 at the band's input (layer 46), from a run that differs in one variable |
| S854-2 | Four row groups (question tail, B1/B4, menu, other boundary rows), with Shapley shares over the groups |

## Hiding images from one route

- **Full attention.** `lib/c_tools.py` `FullAttn` with the `ko` edit masks the hidden positions' keys for the chosen text rows, at every
  full-attention layer. The image rows themselves are untouched, and block shares are read against the unmasked image keys (S748, S703).
- **Gated DeltaNet.** `lib/c_hide.py` `HideDN` recomputes the text rows' outputs with β = 0 and g = 0 at the hidden positions, so the
  recurrent state passes them unchanged, with no write and no decay. The first three text rows' convolution windows take the inputs of the
  three positions before the hidden span (S703).

## Cutting a source through every route (`lib/c_cut.py`, `SourceCut`)

The source is the token span from image N−1's opening marker to image N's closing marker. A cut at layer l removes the span's route at that
layer:

- **Full attention.** Every sample of a batch gets an explicit causal boolean mask. In a cut sample, rows after the span may not attend to it.
  Cut and uncut samples therefore share one SDPA kernel.
- **DeltaNet state.** `torch_chunk_gated_delta_rule` is wrapped to run in three segments: before, over and after the span. In a cut layer,
  the rows after the span start from the state reached before it. This is exact. A large finite decay at the span instead leaks earlier
  content through the in-chunk cumulative sum.
- **DeltaNet convolution.** `causal_conv1d_fn` is wrapped so that, in a cut sample, the outputs of the first three rows after the span are
  recomputed from a spliced window: the three tokens before the span, followed by these rows.

The checks are:

- On random tensors, the cut output equals that of the sequence without the span within 2e-7 (convolution 1.5e-8).
- In the model, swapping the two cut images leaves the cut run bit-identical.
- With the instrument on and nothing cut, answers agree with the native kernels in 96–98% of runs.

S857 uses the cut by layer window and route, and S858 uses it for predictions fixed in advance.

## Rotary geometry

S855 shifts the text rows' multimodal rotary positions by the advance of two images (`D`). This separates where the question sits from what
the extra images contain.

## Exactness controls used throughout

- Reference runs pass through the editing code with the recipient as its own donor. Bit-identity is required where batch shapes allow.
- Donor and recipient runs share batch shapes, because injected activations carry the numerics of their batch shape. This is why S856 was
  re-run as S856b.
- Resident model workers reload the experiment modules before every job (`lib/w27b_worker.py`).
- An invalid run in which a stale module made two arms identical is kept and labelled.
