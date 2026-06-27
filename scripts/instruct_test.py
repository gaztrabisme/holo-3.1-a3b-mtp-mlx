#!/usr/bin/env python3
"""Does the Instruct/non-thinking preset (temp 0.7 + presence_penalty 1.5) preserve grounding accuracy
and coordinate determinism vs greedy? Measures both."""
import json, re, urllib.request, urllib.error, os
HERE=os.path.dirname(__file__)
GT=json.load(open(f"{HERE}/ui_truth.json")); W,H=GT["size"]; TRUTH=GT["truth"]
B64=open(f"{HERE}/ui.b64").read()
RF={"type":"json_schema","json_schema":{"name":"point","schema":{"type":"object","additionalProperties":False,
    "required":["x","y"],"properties":{"x":{"type":"integer"},"y":{"type":"integer"}}}}}
PROMPT=("Localize an element on the GUI image according to the provided target and output a click position.\n"
        " * You must output a valid JSON following the format: {\"x\":int,\"y\":int}\n Your target is:\n{el}")
MODEL="Holo-3.1-35B-A3B-oQ8-mtp"

PROFILES={
 "greedy (temp0)":          {"temperature":0},
 "instruct (t0.7+pp1.5)":   {"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0},
 "instruct NO penalty":     {"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0},
}

def call(el, samp):
    content=[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{B64}"}},
             {"type":"text","text":PROMPT.replace("{el}",el)}]
    body=dict(model=MODEL,messages=[{"role":"user","content":content}],max_tokens=48,
              chat_template_kwargs={"enable_thinking":False},response_format=RF,**samp)
    req=urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",json.dumps(body).encode(),
        {"Authorization":"Bearer 73067799","Content-Type":"application/json"})
    try: d=json.load(urllib.request.urlopen(req,timeout=120))
    except urllib.error.HTTPError as e: return None
    t=d["choices"][0]["message"]["content"]
    try: j=json.loads(t); return j["x"]/1000*W, j["y"]/1000*H
    except Exception:
        n=[float(x) for x in re.findall(r"-?\d+\.?\d*",t)]; return (n[0]/1000*W,n[1]/1000*H) if len(n)>=2 else None

for name,samp in PROFILES.items():
    errs=[]
    for el,(gx,gy) in TRUTH.items():
        p=call(el,samp)
        if p: errs.append(((p[0]-gx)**2+(p[1]-gy)**2)**.5)
    hits=sum(e<70 for e in errs); med=sorted(errs)[len(errs)//2] if errs else -1
    print(f"{name:26} -> {hits}/{len(TRUTH)} hit, median {med:.0f}px, max {max(errs):.0f}px")

# determinism: same target x5 under instruct preset
print("\ndeterminism — 'red Delete Account button' x5 (gt=1171,744):")
for i in range(5):
    p=call("the red Delete Account button", PROFILES["instruct (t0.7+pp1.5)"])
    print(f"  run{i+1}: ({p[0]:.0f},{p[1]:.0f})" if p else "  run unparseable")
