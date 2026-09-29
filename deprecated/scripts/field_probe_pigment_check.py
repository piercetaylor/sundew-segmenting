"""Are the sundew pixels the model MISSES redder than the ones it FINDS?

Both populations are ground-truth sundew, so this compares like with like: it
needs no colour model of 'sundew' and cannot be fooled by sand or leaf litter
the way the discarded YCbCr skin detector was. The only claim is a contrast
between two labelled sets inside the same image.
"""
import json, pathlib, sys
import numpy as np
from PIL import Image
sys.path.insert(0, 'src')
from dataclasses import replace
import torch
from sundew_segmentation.baseline import SEGFORMER_B0, resolve_model

ROOT = pathlib.Path('/cluster/VAST/mendozacozatld-lab/PierceTaylor')
ck = torch.load('models/recipe/combined/seed-17/segformer-b0-best.pt', map_location='cpu', weights_only=True)
cfg = replace(SEGFORMER_B0, **{k: ck['config'][k] for k in SEGFORMER_B0.to_dict() if k in ck['config']})
model = resolve_model(cfg, pretrained=False); model.load_state_dict(ck['model_state']); model.eval()
mean = np.asarray((0.485,0.456,0.406), np.float32)[:,None,None]
std = np.asarray((0.229,0.224,0.225), np.float32)[:,None,None]

rows=[]
for split in ('field-eval','field-train'):
    for imp in sorted((ROOT/'field-probe-eval/images'/split).glob('*.jpg')):
        mp = ROOT/'field-probe-v1/masks'/split/(imp.stem+'.png')
        with Image.open(imp) as s:
            im = s.convert('RGB').resize((cfg.image_size,)*2, Image.Resampling.BILINEAR)
        with Image.open(mp) as s:
            gt = np.asarray(s.convert('L').resize((cfg.image_size,)*2, Image.Resampling.NEAREST))>0
        a = np.asarray(im, np.float32).transpose(2,0,1)/255.
        with torch.no_grad():
            pr = torch.sigmoid(model(torch.from_numpy(((a-mean)/std).copy()).unsqueeze(0)))[0,0].numpy()>=0.5
        rgb = np.asarray(im, np.float32)
        R,G,B = rgb[...,0], rgb[...,1], rgb[...,2]
        redness = (R-G)/(R+G+1e-6)          # >0 = red-dominant, <0 = green-dominant
        found  = gt & pr                     # labelled sundew the model got
        missed = gt & ~pr                    # labelled sundew the model lost
        if found.sum()<500 or missed.sum()<500: continue
        rows.append({'name':imp.name,'split':split,
                     'redness_found':float(redness[found].mean()),
                     'redness_missed':float(redness[missed].mean()),
                     'frac_missed_reddominant':float((redness[missed]>0).mean()),
                     'frac_found_reddominant':float((redness[found]>0).mean()),
                     'missed_px_frac':float(missed.sum()/gt.sum())})
        print(f"{imp.name:<22} {split:<12} found {rows[-1]['redness_found']:+.3f}  missed {rows[-1]['redness_missed']:+.3f}  delta {rows[-1]['redness_missed']-rows[-1]['redness_found']:+.3f}", flush=True)

d = [r['redness_missed']-r['redness_found'] for r in rows]
n = len(d); m = float(np.mean(d)); sd = float(np.std(d, ddof=1))
print(f"\n=== {n} images with both populations >=500 px ===")
print(f"missed-minus-found redness: mean {m:+.4f}  SD {sd:.4f}  SE {sd/np.sqrt(n):.4f}")
print(f"  t = {m/(sd/np.sqrt(n)):.2f} on {n-1} df")
print(f"  images where missed tissue is redder than found tissue: {sum(1 for x in d if x>0)}/{n}")
print(f"mean fraction of MISSED sundew px that are red-dominant: {np.mean([r['frac_missed_reddominant'] for r in rows]):.3f}")
print(f"mean fraction of FOUND  sundew px that are red-dominant: {np.mean([r['frac_found_reddominant'] for r in rows]):.3f}")
json.dump(rows, open('data/reports/field-probe-evaluation/pigment-analysis.json','w'), indent=2)
