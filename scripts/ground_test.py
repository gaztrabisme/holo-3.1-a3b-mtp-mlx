#!/usr/bin/env python3
"""Holo-3.1 grounding spot-check vs synthetic UI ground truth.
Official prompt (hub.hcompany.ai/element-localization): normalized [0,1000] {x,y} JSON."""
import json, re, sys, urllib.request, urllib.error, os

URL="http://127.0.0.1:8000/v1/chat/completions"; KEY="73067799"
HERE=os.path.dirname(__file__)
GT=json.load(open(f"{HERE}/ui_truth.json")); W,H=GT["size"]; TRUTH=GT["truth"]
B64=open(f"{HERE}/ui.b64").read()

SCHEMA={"properties":{"x":{"description":"X coordinate as integer in [0, 1000]","ge":0,"le":1000,"title":"X","type":"integer"},
                     "y":{"description":"Y coordinate as integer in [0, 1000]","ge":0,"le":1000,"title":"Y","type":"integer"}},
        "required":["x","y"],"title":"VisualLocalizerOutput","type":"object"}
PROMPT=("Localize an element on the GUI image according to the provided target and output a click position.\n"
        " * You must output a valid JSON following the format: {schema}\n"
        " Your target is:\n{element}")
# response_format schema (oMLX json_schema enforcement) — drop H-specific ge/le (not standard JSON Schema)
RF={"type":"json_schema","json_schema":{"name":"point","schema":{
    "type":"object","additionalProperties":False,"required":["x","y"],
    "properties":{"x":{"type":"integer"},"y":{"type":"integer"}}}}}

def call(model, element, max_tokens=48, enforce=True):
    txt=PROMPT.format(schema=json.dumps(SCHEMA), element=element)
    content=[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{B64}"}},
             {"type":"text","text":txt}]
    payload={"model":model,"messages":[{"role":"user","content":content}],
             "max_tokens":max_tokens,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    if enforce: payload["response_format"]=RF
    body=json.dumps(payload).encode()
    req=urllib.request.Request(URL,body,{"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    try: d=json.load(urllib.request.urlopen(req,timeout=120))
    except urllib.error.HTTPError as ex: return f"<HTTP {ex.code}: {ex.read().decode()[:140]}>"
    return d["choices"][0]["message"]["content"]

def parse(txt):
    try:
        j=json.loads(txt); return float(j["x"]), float(j["y"])
    except Exception:
        n=[float(x) for x in re.findall(r"-?\d+\.?\d*",txt)]
        return (n[0],n[1]) if len(n)>=2 else None

def run(model):
    print(f"\n=== grounding spot-check: {model} ===")
    outs={}; errs=[]
    for el,(gx,gy) in TRUTH.items():
        raw=call(model,el); xy=parse(raw); outs[el]=raw
        if xy:
            px,py=xy[0]/1000*W, xy[1]/1000*H
            err=((px-gx)**2+(py-gy)**2)**.5; errs.append(err)
            tag="HIT" if err<70 else ("near" if err<140 else "MISS")
            print(f"  {el[:46]:48} gt=({gx:4d},{gy:3d}) -> ({px:4.0f},{py:3.0f}) err={err:3.0f}px {tag}")
        else:
            print(f"  {el[:46]:48} gt=({gx:4d},{gy:3d}) -> UNPARSEABLE {raw[:50]!r}")
    if errs:
        hits=sum(e<70 for e in errs); near=sum(70<=e<140 for e in errs)
        print(f"  --> {hits}/{len(errs)} HIT (<70px), {near} near, median {sorted(errs)[len(errs)//2]:.0f}px, max {max(errs):.0f}px")
    return outs

if __name__=="__main__":
    model=sys.argv[1] if len(sys.argv)>1 else "Holo-3.1-35B-A3B-oQ8-mtp"
    outs=run(model)
    json.dump(outs, open(f"{HERE}/ground_{model.split('/')[-1]}.json","w"), indent=2)
