# holo-3.1-a3b-mtp-mlx

Grafting a **Multi-Token-Prediction (MTP / nextn) head** onto the 8-bit MLX build of **[Holo-3.1-35B-A3B](https://huggingface.co/Hcompany/Holo-3.1-35B-A3B)** — H Company's GUI-grounding / computer-use VLM — so [oMLX](https://github.com/jundot/omlx) can run native speculative decoding.

🤗 **Model:** https://huggingface.co/GazTrab/Holo-3.1-35B-A3B-oQ8-mtp

## TL;DR

- **No Holo-3.1 release ships an MTP head** — H Company stripped the nextn layer when fine-tuning (verified across bf16 / oQ8 / GGUF / NVFP4). It can only come from the `Qwen3.6-35B-A3B` base that Holo was fine-tuned from.
- Holo-3.1 and that base share a **byte-identical non-MTP tensor set** (`qwen3_5_moe`, 2010 tensors, hidden 2048, 40 layers, 256 experts, vocab 248320), so the 42-tensor 8-bit `language_model.mtp.*` block from [`tfjack/Qwen3.6-35B-A3B-oQ8-fp16-mtp`](https://huggingface.co/tfjack/Qwen3.6-35B-A3B-oQ8-fp16-mtp) drops straight into [`symrex/Holo-3.1-35B-A3B-oQ8`](https://huggingface.co/symrex/Holo-3.1-35B-A3B-oQ8) with **no transforms** (already 8-bit, group-size matched, oMLX-formatted).
- **Lossless** (spec-decode verifies every token → output identical to base) and **free** (local file surgery, no retrain, no download).

## Results (M3 Max 96 GB, single-stream, greedy)

| | base oQ8 | **oQ8-mtp** |
|---|---|---|
| Decode (300 tok) | 55.0 tok/s | **60.5 tok/s (~1.1×)** |
| MTP acceptance (structured text) | — | **94.8%** (145/153) |
| Output vs base | — | **byte-identical** |
| Grounding (12-target synthetic UI) | 12/12 hit, med 6 px | **12/12 hit, med 6 px, identical coords** |

The ~1.1× (not ~2×, despite 94.8% acceptance) is honest: a 1-layer MTP caps at ~2 tok/cycle, and at batch=1 the 2-position verify pass on this memory-bound MoE isn't much cheaper than two plain passes. MTP earns more on **longer agentic/navigation traces** than on short bbox grounding, where screenshot *prefill* dominates latency.

### Grounding spot-check

A synthetic 1280×720 settings UI with 12 elements at known positions. Holo was asked to localize each (e.g. *"the red Delete Account button"*); the figure overlays **ground truth** (green rings) against **Holo's predicted click** (red crosshairs). All 12 land on-target; the two visible offsets are the search box (~14 px) and the wide password field (55 px, still inside it). `scripts/make_ui.py` builds the fixture, `scripts/ground_test.py` scores it, `scripts/annotate_ui.py` renders this overlay.

![ground truth vs Holo predicted click points — 12/12 hit, median 6px](assets/ui_annotated.png)

## Reproduce

```bash
# 1. graft the MTP block (reads symrex + tfjack from ~/.omlx/models, writes the new model dir)
python scripts/graft_mtp.py

# 2. restart oMLX so it discovers the model, then benchmark decode speed
python scripts/bench_one.py Holo-3.1-35B-A3B-oQ8-mtp out_mtp.txt
python scripts/bench_one.py Holo-3.1-35B-A3B-oQ8     out_base.txt   # losslessness: diff the two

# 3. grounding accuracy vs ground truth
python scripts/make_ui.py            # synthetic UI + ground-truth coords
python scripts/ground_test.py Holo-3.1-35B-A3B-oQ8-mtp
```

`accept=N/M` and `MTP path activated` show up in `~/.omlx/logs/server.log`.

## What's here

| Path | |
|---|---|
| `scripts/graft_mtp.py` | the graft: 42 `mtp.*` tensors + config flags + index rebuild |
| `scripts/bench_one.py` | single-model decode benchmark (isolated, no dual residency) |
| `scripts/make_ui.py` | synthetic UI screenshot with known element positions |
| `scripts/ground_test.py` | grounding accuracy vs ground truth (official Holo-3.1 prompt) |
| `docs/design.md` | full design + research notes (lineage, MTP mechanics, sampling) |
| `results/` | grounding outputs (base + mtp) |

## Grounding recipe (Holo-3.1)

Single user turn, image + text, output **normalized `[0,1000]` JSON `{x,y}`** (not Holo1's pixel `Click(x,y)`). `temperature=0`, `enable_thinking=False`, enforce shape with `response_format` json-schema. **smart_resize** the screenshot first and scale coords against the dimensions you send. Full example in the [model card](https://huggingface.co/GazTrab/Holo-3.1-35B-A3B-oQ8-mtp).

## Attribution

Apache-2.0. Stands on **H Company** (Holo-3.1), **Qwen** (Qwen3.6-35B-A3B base + MTP head), **symrex** (oQ8 quant), **tfjack** (Qwen3.6 oQ8 MTP MLX conversion), and **oMLX / oQ** (quantization + serving).
