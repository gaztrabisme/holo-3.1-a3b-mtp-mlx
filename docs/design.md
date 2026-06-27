# Holo-3.1-35B-A3B → oMLX 8-bit + MTP — Design

**Goal:** Produce `Holo-3.1-35B-A3B-oQ8-mtp` — the existing 8-bit Holo grounding model with a working MTP (nextn) head so oMLX can do native speculative decoding.

## STATUS: BUILT, VERIFIED & PUBLISHED (2026-06-27)

- **HF model:** https://huggingface.co/GazTrab/Holo-3.1-35B-A3B-oQ8-mtp (public, Apache-2.0)
- **GitHub (code+docs):** https://github.com/gaztrabisme/holo-3.1-a3b-mtp-mlx (public)


Model at `~/.omlx/models/symrex/Holo-3.1-35B-A3B-oQ8-mtp`, served by oMLX as `Holo-3.1-35B-A3B-oQ8-mtp`.

Measured on this M3 Max (96 GB), single-stream, greedy, structured low-entropy output (50-item JSON, 300 tok):
- **MTP engages:** oMLX log `MTP path activated ... accept=145/153 (94.8%)`. Acceptance **94.8%** — far above the
  5–46% the research warned about for quantized MoE MTP; confirms the low-entropy/greedy best-case thesis.
- **Lossless:** base vs mtp greedy output **byte-identical** (sha1 `17981a3178ca`, len 504). MTP changes speed only,
  not output — bbox coords unaffected.
- **Decode speed:** base **55.0** tok/s → mtp **60.5** tok/s = **~1.1x** single-stream. Modest: 1-layer MTP caps at
  ~2 tok/cycle, and at batch=1 the 2-position verify pass on this memory-bound MoE isn't much cheaper than 2 single
  passes (backbone ≈3.9 s, mtp head ≈0.64 s overhead per 300 tok). Speedup grows with longer generations; marginal
  for very short bbox-only outputs.
- **Memory:** 36.4 GB resident. Two 38 GB models can't coexist (70 GB prefill guard); oMLX LRU auto-evicts the idle
  one — safe as long as you don't hammer both concurrently. Do NOT use `omlx restart` (collides with the GUI app);
  the app respawns its own server.

**Verdict:** free, lossless, correct. Worth keeping for medium/long generations and feature parity; benefit is small
for pure short-coordinate grounding.

### Vision-path spot-check (synthetic 1280×800 UI, 12 known targets)
- **Accuracy: 12/12 HIT (<70px), median err 6px, max 55px.** Holo nails every element. (Worst = wide password field,
  model clicked left-of-center but inside it.) Base and MTP both 12/12.
- **Vision path LOSSLESS: 12/12 coords byte-identical** base vs mtp. Spec-decode verification holds on the VLM path.
- **MTP engages on vision** (`MTP path activated ... accept=6/10 (60%)`), but acceptance is lower (33–60%) because
  grounding outputs are tiny (~6–18 tokens of `{x,y}`). For pure grounding, screenshot **prefill dominates latency**,
  not the ~15-token decode → MTP's speed benefit is negligible here. MTP pays off on longer agentic/navigation traces.

### Holo-3.1 grounding recipe (official, hub.hcompany.ai/element-localization)
- Single user turn (no system), image + text. Output = normalized **[0,1000]** JSON `{"x":int,"y":int}` (NOT Holo1's
  pixel `Click(x,y)`). Scale: `px = x/1000*W`, `py = y/1000*H`, origin top-left.
- Prompt: `Localize an element on the GUI image according to the provided target and output a click position.\n * You must output a valid JSON following the format: {schema}\n Your target is:\n{element}`
- `temperature=0`, `enable_thinking=False`; enforce shape with `response_format` json_schema `{x,y}` for clean parse.
- **smart_resize**: pre-resize image to Qwen factor (patch16·merge2=32), min 65536 / max 16.7M px, so server-side
  resize doesn't shift coords. (1280×800 is already 32-divisible + in-bounds → no-op; arbitrary screenshots need it.)
- Harness: `scratchpad/{make_ui.py, ground_test.py}` (reusable).

## Verified facts (local + web, 2026-06-27)

- Both local models share **byte-identical non-MTP tensor sets** (2010 tensors each), identical dims
  (hidden 2048, 40 layers, 256 experts/8 active, head_dim 256, 2 KV heads), identical tokenizer
  (vocab 248320, bos 248044). `model_type: qwen3_5_moe`, arch `Qwen3_5MoeForConditionalGeneration`.
  → Same architecture; MTP block fits Holo exactly.
- **Version label 3.5 vs 3.6:** Hcompany Holo card frontmatter `base_model[0] = Qwen/Qwen3.6-35B-A3B`;
  prose says "Qwen 3.5 family"; the *smaller* Holo-3.1 siblings (9B/4B/0.8B) are genuinely Qwen3.5.
  So the 35B is on 3.6, siblings on 3.5 — hence the mixed signal. Arch class is named `qwen3_5_moe`
  for both 3.5 and 3.6 (shared architecture, version is a checkpoint increment). Donor `tfjack` is also
  Qwen3.6-35B-A3B → **same lineage as Holo's base.**
- **No Holo release ships MTP.** Hcompany bf16, symrex oQ8, Hcompany GGUF, Hcompany NVFP4 — all strip the
  nextn head (H Company removed it when fine-tuning). MTP can ONLY come from the Qwen3.6 base. There is no
  "reconvert from source and keep MTP" path. Grafting is the only option (short of training an MTP head).
- **Donor MTP = 42 tensors** under `language_model.mtp.*` in `tfjack/Qwen3.6-35B-A3B-oQ8-fp16-mtp`:
  `mtp.fc.weight (2048,4096)`, `mtp.pre_fc_norm_embedding/hidden.weight`, `mtp.norm.weight`, and one full
  MoE decoder layer `mtp.layers.0.*` (self_attn q/k/v/o + q/k_norm, switch_mlp 256 experts, shared_expert, gate).
  Linears are **8-bit** (carry `.scales`/`.biases`); norms unquantized. → matches Holo backbone quant (8-bit affine gs64).
- **Config enable flags** (present in donor, absent in Holo): `text_config.mtp_num_hidden_layers = 1`,
  `text_config.mtp_use_dedicated_embeddings = False` (MTP reuses main embed_tokens + lm_head). Plus 6 MTP
  per-module entries in the `quantization` dict.
- oMLX MTP naming = mlx-lm PR#990 scheme (`mtp.fc`, `mtp.layers.*`, `pre_fc_norm_*`), NOT the HF `eh_proj/enorm/hnorm`
  scheme. Because we graft from an already-oMLX-converted sibling, **no RMSNorm +1 offset and no expert re-stack
  needed** — tensors are already in oMLX's consumed form. Drop-in.

## Plan (Option A — graft, all local, 0 download)

1. Copy `symrex/Holo-3.1-35B-A3B-oQ8` → `Holo-3.1-35B-A3B-oQ8-mtp` (~38 GB).
2. Read the 42 `language_model.mtp.*` tensors from the donor; write them into a new shard (or the last shard)
   of the copy; rebuild `model.safetensors.index.json`.
3. Merge into the copy's `config.json`: `text_config.mtp_num_hidden_layers=1`,
   `text_config.mtp_use_dedicated_embeddings=False`, and the 6 MTP quant-override entries (mirror into
   `quantization_config` too).
4. Load in oMLX; confirm it engages the nextn head; benchmark **acceptance rate** + tok/s vs current Holo-oQ8,
   on a grounding prompt (screenshot → bbox), greedy.

## Risk & fallback

- Correctness is **guaranteed** (spec decode verifies every token → output == Holo's exact greedy output).
  Only acceptance rate (speedup magnitude) is at stake.
- Donor MTP trained on *base* Qwen3.6 hidden states; Holo fine-tune drifted them → acceptance below a native MTP.
  **Grounding is the best case:** low-entropy bbox coordinates + greedy decode = highest spec-decode acceptance.
- Quant-matched (8-bit MTP on 8-bit backbone) avoids the BF16-MTP-on-quant-backbone "0% acceptance" dtype trap.
- **Fallback (only if measured acceptance is poor):** extract fp16 MTP from Qwen3.6 base bf16 (apply +1 norm
  offset, re-stack experts), keep fp16 per the `oQ8-fp16-mtp` recipe. Costs ~70 GB base download.

## Sampling for grounding

- Holo's shipped `generation_config` (do_sample, temp 1.0, top_k 20, top_p 0.95) = inherited generic Qwen chat
  default — **wrong for bbox grounding.**
- Use **greedy: temperature 0 / do_sample off** for grounding/localization. Coordinates output in normalized
  [0,1000] space; sampling jitters clicks and breaks reproducibility. (H Company publishes no official grounding
  sampling table; greedy-for-grounding is standard convention + community notes.)
- Disable thinking (`chat_template_kwargs.enable_thinking=False`) for direct coordinate output.
- **Two-config model (measured 2026-06-27):** greedy = 12/12 hit, median 3px, deterministic. Instruct preset
  (temp 0.7 / top_p 0.8 / top_k 20 / min_p 0 / presence_penalty 1.5) = 11/12, median 27px, ±40px jitter — the
  `presence_penalty` skews the short `{x,y}` numeric output. So: **grounding → greedy temp 0 no penalties**;
  **agentic/navigation/general → instruct preset** (Qwen3 anti-loop, and where MTP acceptance climbs **>90%**).
- **CORRECTION:** "greedy maximizes MTP acceptance" is FALSE as a general rule. It holds for low-entropy text
  (94.8% greedy on JSON), but at temp>0 oMLX uses probabilistic rejection sampling `min(1, p_t/p_d)` which is more
  lenient than greedy's argmax match → sampling can accept MORE drafts (the >90% finding). Acceptance affects speed
  only, never output. On grounding the split is harmless: outputs are ~15 tok (prefill-dominated), so greedy's lower
  acceptance costs no wall-clock, and `generation_config.json` ships the instruct preset with a temp-0 grounding override.
