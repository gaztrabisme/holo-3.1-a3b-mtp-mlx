#!/usr/bin/env python3
"""Synthetic UI screenshot with KNOWN element positions, for grounding ground-truth."""
import json, os
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 800
img = Image.new("RGB", (W, H), (244, 245, 248))
d = ImageDraw.Draw(img)

def font(sz, bold=False):
    for p in ([f"/System/Library/Fonts/Helvetica.ttc"] if not bold else
              ["/System/Library/Fonts/Helvetica.ttc"]):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()

F  = font(22); FB = font(26); FS = font(18)

def center(box): return ((box[0]+box[2])//2, (box[1]+box[3])//2)
def btn(box, label, fill, fg=(255,255,255), f=F):
    d.rounded_rectangle(box, radius=10, fill=fill)
    tb = d.textbbox((0,0), label, font=f); tw, th = tb[2]-tb[0], tb[3]-tb[1]
    cx, cy = center(box)
    d.text((cx-tw/2, cy-th/2-tb[1]+tb[1]), label, font=f, fill=fg)
    return label, center(box), box

truth = {}

# top bar
d.rectangle([0,0,W,64], fill=(255,255,255)); d.line([0,64,W,64], fill=(225,227,232), width=2)
d.text((28,20), "Account Settings", font=FB, fill=(30,32,38))
# search box (top right)
sb=[920,16,1130,48]; d.rounded_rectangle(sb, radius=8, outline=(200,202,208), width=2, fill=(248,249,251))
d.text((sb[0]+12, sb[1]+6), "Search…", font=FS, fill=(150,152,158))
truth["the search box"]=center(sb)
# avatar (top far right)
av=[1170,12,1222,52]; d.ellipse(av, fill=(90,120,220)); d.text((av[0]+16,av[1]+8),"GT",font=F,fill=(255,255,255))
truth["the profile avatar (top-right circle)"]=center(av)

# left sidebar menu
for i,(lab) in enumerate(["Profile","Security","Billing","Notifications"]):
    y=110+i*54; box=[20,y,240,y+44]
    sel = (lab=="Security")
    if sel: d.rounded_rectangle(box, radius=8, fill=(232,238,255))
    d.text((40,y+10), lab, font=F, fill=(40,80,210) if sel else (70,72,80))
    truth[f"the '{lab}' menu item in the left sidebar"]=center(box)

# main form
d.text((300,110), "Security", font=FB, fill=(30,32,38))
# a labeled input
d.text((300,170),"Current password", font=FS, fill=(110,112,120))
ib=[300,195,760,235]; d.rounded_rectangle(ib, radius=8, outline=(200,202,208), width=2, fill=(255,255,255))
d.text((312,205),"••••••••", font=F, fill=(80,82,90)); truth["the current password input field"]=center(ib)
# a checkbox
cb=[300,270,326,296]; d.rounded_rectangle(cb, radius=5, outline=(160,162,170), width=2, fill=(255,255,255))
d.text((338,272),"Enable two-factor authentication", font=F, fill=(60,62,70))
truth["the two-factor authentication checkbox"]=center(cb)
# toggle (drawn as pill)
tg=[300,330,360,360]; d.rounded_rectangle(tg, radius=15, fill=(60,190,120)); d.ellipse([334,332,358,358],fill=(255,255,255))
d.text((376,335),"Login alerts", font=F, fill=(60,62,70)); truth["the 'Login alerts' toggle switch"]=center(tg)

# bottom action buttons
b_cancel = btn([300,720,430,768], "Cancel", (235,236,240), (70,72,80))
truth["the Cancel button"]=center([300,720,430,768])
b_save = btn([450,720,610,768], "Save Changes", (40,120,240))
truth["the Save Changes button"]=center([450,720,610,768])
b_delete = btn([1090,720,1252,768], "Delete Account", (220,60,60))
truth["the red Delete Account button"]=center([1090,720,1252,768])

out=os.path.join(os.path.dirname(__file__),"ui.png")
img.save(out)
json.dump({"size":[W,H],"truth":truth}, open(out.replace(".png","_truth.json"),"w"), indent=2)
print("saved", out, f"({W}x{H})")
print(f"{len(truth)} ground-truth targets:")
for k,v in truth.items(): print(f"  {v}  <- {k}")
