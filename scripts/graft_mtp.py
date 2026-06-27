#!/usr/bin/env python3
"""Graft the Qwen3.6-35B-A3B MTP (nextn) head onto symrex Holo-3.1-35B-A3B-oQ8.
All local, 8-bit-on-8-bit (quant-matched), no transforms (donor already oMLX-formatted)."""
import json, os, shutil, glob, sys
import mlx.core as mx

HOME = os.path.expanduser("~")
SRC_HOLO  = f"{HOME}/.omlx/models/symrex/Holo-3.1-35B-A3B-oQ8"
SRC_DONOR = f"{HOME}/.omlx/models/tfjack/Qwen3.6-35B-A3B-oQ8-fp16-mtp"
DST       = f"{HOME}/.omlx/models/symrex/Holo-3.1-35B-A3B-oQ8-mtp"

ST_DSIZE = {"F64":8,"F32":4,"F16":2,"BF16":2,"I64":8,"U64":8,"I32":4,"U32":4,
            "I16":2,"U16":2,"I8":1,"U8":1,"BOOL":1}

def shard_nbytes(path):
    """sum tensor nbytes from a safetensors header without loading data."""
    from safetensors import safe_open
    tot = 0
    with safe_open(path, framework="np") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            n = 1
            for d in sl.get_shape(): n *= d
            tot += n * ST_DSIZE[sl.get_dtype()]
    return tot

def main():
    os.makedirs(DST, exist_ok=True)

    # ---- 1. extract 42 MTP tensors from donor ----
    donor_map = json.load(open(glob.glob(SRC_DONOR + "/*index*.json")[0]))["weight_map"]
    mtp_keys = [k for k in donor_map if k.startswith("language_model.mtp")]
    shards = sorted(set(donor_map[k] for k in mtp_keys))
    mtp = {}
    for sh in shards:
        w = mx.load(os.path.join(SRC_DONOR, sh))
        for k in mtp_keys:
            if donor_map[k] == sh:
                mtp[k] = w[k]
    print(f"[1] extracted {len(mtp)} MTP tensors from donor shards {shards}")
    assert len(mtp) == len(mtp_keys) == 42, f"expected 42, got {len(mtp)}"

    # ---- 2. rewrite Holo's last shard = its tensors + MTP ----
    holo_idx_path = glob.glob(SRC_HOLO + "/*index*.json")[0]
    holo_idx = json.load(open(holo_idx_path))
    holo_map = holo_idx["weight_map"]
    shard_files = sorted(set(holo_map.values()))
    last = shard_files[-1]
    last_w = mx.load(os.path.join(SRC_HOLO, last))
    n_before = len(last_w)
    merged = dict(last_w); merged.update(mtp)
    mx.save_safetensors(os.path.join(DST, last), merged, metadata={"format": "mlx"})
    print(f"[2] rewrote {last}: {n_before} + {len(mtp)} mtp = {len(merged)} tensors")

    # ---- 3. hardlink unchanged shards + copy aux files ----
    for sh in shard_files[:-1]:
        dst = os.path.join(DST, sh)
        if os.path.exists(dst): os.remove(dst)
        os.link(os.path.join(SRC_HOLO, sh), dst)
    aux = ["tokenizer.json","tokenizer_config.json","vocab.json","merges.txt",
           "chat_template.jinja","preprocessor_config.json","generation_config.json",".gitattributes"]
    for a in aux:
        s = os.path.join(SRC_HOLO, a)
        if os.path.exists(s): shutil.copy2(s, os.path.join(DST, a))
    print(f"[3] hardlinked {len(shard_files)-1} shards + copied aux")

    # ---- 4. config.json: enable flags + 6 MTP quant overrides ----
    cfg = json.load(open(os.path.join(SRC_HOLO, "config.json")))
    cfg["text_config"]["mtp_num_hidden_layers"] = 1
    cfg["text_config"]["mtp_use_dedicated_embeddings"] = False
    dcfg = json.load(open(os.path.join(SRC_DONOR, "config.json")))
    added = 0
    for qk in ("quantization", "quantization_config"):
        if qk in dcfg and qk in cfg:
            for k, v in dcfg[qk].items():
                if isinstance(v, dict) and k.startswith("language_model.mtp"):
                    cfg[qk][k] = v; added += 1
    json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
    print(f"[4] config: set mtp flags + merged {added} mtp quant overrides")

    # ---- 5. rebuild index.json ----
    new_map = dict(holo_map)
    for k in mtp_keys: new_map[k] = last
    total = sum(shard_nbytes(os.path.join(DST, sh)) for sh in shard_files)
    json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
              open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=2)
    print(f"[5] index: {len(new_map)} tensors, total_size={total/1e9:.2f} GB")

    # ---- verify ----
    vi = json.load(open(os.path.join(DST, "model.safetensors.index.json")))["weight_map"]
    vc = json.load(open(os.path.join(DST, "config.json")))
    nmtp = sum(1 for k in vi if "mtp" in k.lower())
    print(f"\n[verify] mtp tensors in index: {nmtp} | "
          f"mtp_num_hidden_layers={vc['text_config'].get('mtp_num_hidden_layers')} | "
          f"dedicated_emb={vc['text_config'].get('mtp_use_dedicated_embeddings')}")
    print(f"[verify] DST = {DST}")
    print("OK" if nmtp == 42 else "FAIL")

if __name__ == "__main__":
    main()
