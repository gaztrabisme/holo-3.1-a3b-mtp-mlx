#!/usr/bin/env python3
import json, urllib.request, urllib.error, sys, hashlib

URL="http://127.0.0.1:8000/v1/chat/completions"; KEY="73067799"
PROMPT=("Output a JSON array of 50 objects, each {\"i\": n, \"sq\": n*n} for n=1..50. "
        "Output only compact JSON, no prose, no whitespace.")
model=sys.argv[1]; out_path=sys.argv[2]

def call(max_tokens=300):
    body=json.dumps({"model":model,"messages":[{"role":"user","content":PROMPT}],
        "max_tokens":max_tokens,"temperature":0,
        "chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request(URL,body,{"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    try:
        d=json.load(urllib.request.urlopen(req,timeout=300))
    except urllib.error.HTTPError as e:
        print("HTTP",e.code,e.read().decode()[:200]); raise
    u=d["usage"]; return u["output_tokens"],u.get("total_time"),u.get("model_load_duration",0),d["choices"][0]["message"]["content"]

ot,tt,ld,content=call()                       # warmup/load
print(f"  [load {ld:.1f}s]")
tpss=[]
for i in range(3):
    ot,tt,ld,content=call()
    tps=ot/tt if tt else 0; tpss.append(tps)
    print(f"  run{i+1}: {ot} tok / {tt:.3f}s = {tps:.1f} tok/s")
best=max(tpss)
open(out_path,"w").write(content)
h=hashlib.sha1(content.encode()).hexdigest()[:12]
print(f"  BEST={best:.1f} tok/s | out_sha1={h} | len={len(content)}")
print(f"BEST_TPS={best:.1f}")
