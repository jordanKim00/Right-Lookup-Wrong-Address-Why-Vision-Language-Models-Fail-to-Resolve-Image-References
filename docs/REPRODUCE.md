# Reproducing

## Software

Every run used one container with:

- Python 3.11.10
- PyTorch 2.5.1 (CUDA 12.1)
- Transformers 5.16.1

Weights are loaded in bfloat16 with scaled dot-product attention (`lib/uni.py`).

## Checkpoints

| Model | Checkpoint | Mount used in the scripts | `config.json` SHA-256 |
|---|---|---|---|
| Qwen3-VL-4B | `Qwen/Qwen3-VL-4B-Instruct` | `/dev/shm/vlm_models/Qwen3-VL-4B-Instruct` | `edac77033291` |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | `/dev/shm/vlm_models/Qwen3-VL-8B-Instruct` | `5cd452860dc1` |
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | `/dev/shm/vlm_models/Qwen3.5-4B` | `ddc63e1c717a` |
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` | `/dev/shm/vlm_models/Qwen3.5-9B` or `/models/Qwen3.5-9B` | `d0883072e018` |
| Qwen3.5-27B | `Qwen/Qwen3.5-27B` | `/models/Qwen3.5-27B` | `f8d190c5b89c` |

The last column is the first twelve hex digits of each local snapshot's `config.json` SHA-256.

## Layout

The scripts use absolute paths of the original workspace:

- `/vlm/exp` for modules
- `/vlm/results/r3` for results
- `/vlm/research/ces/outputs` for derived files

To run them from this repository:

```bash
python tools/link_layout.py            # creates exp/, results/ and research/ as symbolic links into this repository
docker run --gpus all -v "$PWD":/vlm -v /path/to/checkpoints:/models -v /dev/shm:/dev/shm <image> \
    python /vlm/exp/s857_analyze.py s857_27b      # re-run a registered analysis on the released records
```

- **Analyses.** Analyses that read only JSON records run on the released data.
- **Experiments.** Re-running an experiment needs a GPU and the checkpoint. The usage line is in each script's docstring, for example
  `python /vlm/exp/s857_cut.py /models/Qwen3.5-27B s857_27b --n 30`.
- **Results.** Results are written to `/vlm/results/r3/`.

## Prompts

A procedural question is one user turn after the system message "You are a helpful assistant.":

- the image placeholders
- a line break
- the question, for example "There are 7 images in a row. What is in the third image from the last?"
- "Answer with only one word from:" followed by the menu words

For Qwen3.5 the assistant turn starts with an empty thinking block (`<think>\n\n</think>\n\n`). For Qwen3-VL Instruct it starts with
nothing.

## Scenes and seeds

- **Procedural pictures** (`lib/r3_proc.py`). Single shapes with one-token labels. They are resized to 160 pixels in the mechanism
  experiments and to 224, 448 or 672 pixels in the image-size test.
- **CIFAR-10 photographs.** The index-code, key–value and transplant experiments use them at 224 pixels.
- **Seeds.** Each experiment fixes its scene seed in its script. Re-runs reuse the original seed, so they see the same scenes with the
  corrected readout.

## Readout switch

`R3_READ=max` (the default of the re-measurements) reads the menu word with the highest logit in either tokenization. `R3_READ=bare` reads the
space-free form.
