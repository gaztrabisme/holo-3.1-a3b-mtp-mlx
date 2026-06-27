#!/usr/bin/env python3
"""Overlay ground-truth + Holo predicted click points on the UI to visualize grounding accuracy."""
import json, os, re
from PIL import Image, ImageDraw, ImageFont

HERE=os.path.dirname(__file__)
GT=json.load(open(f"{HERE}/ui_truth.json")); W,H=GT["size"]; TRUTH=GT["truth"]
PRED=json.load(open(f"{HERE}/ground_Holo-3.1-35B-A3B-oQ8-mtp.json"))
img=Image.open(f"{HERE}/ui.png").convert("RGB"); d=ImageDraw.Draw(img)
def font(s):
    try: return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",s)
    except Exception: return ImageFont.load_default()
F=font(15); FB=font(18)

def parse(t):
    try: j=json.loads(t); return j["x"]/1000*W, j["y"]/1000*H
    except Exception:
        n=[float(x) for x in re.findall(r"-?\d+\.?\d*",t)]; return (n[0]/1000*W,n[1]/1000*H) if len(n)>=2 else None

GREEN=(0,170,80); RED=(225,45,45)
errs=[]
for el,(gx,gy) in TRUTH.items():
    p=parse(PRED[el]);
    if not p: continue
    px,py=p; err=((px-gx)**2+(py-gy)**2)**.5; errs.append(err)
    # GT: green hollow ring
    d.ellipse([gx-9,gy-9,gx+9,gy+9], outline=GREEN, width=3)
    # prediction: red crosshair dot
    d.line([px-8,py,px+8,py], fill=RED, width=3); d.line([px,py-8,px,py+8], fill=RED, width=3)
    d.ellipse([px-3,py-3,px+3,py+3], fill=RED)
    if err>20: d.line([gx,gy,px,py], fill=(120,120,120), width=1)

# legend box
lb=[16,640,250,712]; d.rectangle(lb, fill=(255,255,255), outline=(200,200,200), width=2)
d.ellipse([30,656,46,672], outline=GREEN, width=3); d.text((54,656),"ground truth",font=F,fill=(40,40,40))
d.line([30,690,46,690],fill=RED,width=3); d.line([38,682,38,698],fill=RED,width=3); d.ellipse([35,687,41,693],fill=RED)
d.text((54,684),"Holo prediction",font=F,fill=(40,40,40))
med=sorted(errs)[len(errs)//2]
d.text((300,648),f"12/12 hit · median {med:.0f}px error",font=FB,fill=(20,20,20))

out=f"{HERE}/ui_annotated.png"; img.save(out)
print("saved",out,"| median err",f"{med:.0f}px","| max",f"{max(errs):.0f}px")
