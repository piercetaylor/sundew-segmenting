"""Does the crop arm's advantage come from actually cropping?

The predicted box covers a median 70% of the frame, so on much of the data the
"crop" is nearly the whole picture. If the benefit is real cropping, it should
concentrate on images where the box is small. If it is uniform across box sizes,
something else is doing the work and tightening the segmentation would not help.
"""
import json, pathlib, statistics as st
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

BASE = pathlib.Path('/cluster/VAST/mendozacozatld-lab/PierceTaylor')
SET = BASE / 'sundew-species-set'
SEEDS = (17, 101, 202, 303, 404)

labels = json.loads((SET / 'labels.json').read_text())
idx = {l: i for i, l in enumerate(labels)}
rows = [json.loads(l) for l in open(SET / 'species-records.jsonl') if l.strip()]
val = [r for r in rows if r['split'] == 'validation']
box = {}
for l in open(SET / 'crops/crop-review-manifest.jsonl'):
    m = json.loads(l)
    box[m['name']] = m['box_frac']

tf = transforms.Compose([transforms.Resize(int(224 * 1.14)), transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def score(arm, seed):
    ck = torch.load(f'models/species-compare/{arm}/seed-{seed}/classifier-best.pt',
                    map_location='cpu', weights_only=False)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(labels))
    m.load_state_dict(ck['model_state']); m.eval(); m.to(dev)
    out = {}
    with torch.no_grad():
        for r in val:
            p = (pathlib.Path(r['image']) if arm == 'full'
                 else SET / 'crops/crops' / f"inat_{r['photo_id']}.jpg")
            with Image.open(p) as im:
                x = tf(im.convert('RGB')).unsqueeze(0).to(dev)
            out[r['photo_id']] = int(m(x).argmax(1).item() == idx[r['label']])
    return out


acc = {a: {s: score(a, s) for s in SEEDS} for a in ('full', 'crop')}
print("scored both arms on the validation split\n")

per = []
for r in val:
    pid = r['photo_id']
    f = st.mean(acc['full'][s][pid] for s in SEEDS)
    c = st.mean(acc['crop'][s][pid] for s in SEEDS)
    per.append({'pid': pid, 'label': r['label'], 'full': f, 'crop': c,
                'delta': c - f, 'box': box[f"inat_{pid}.jpg"]})

q = sorted(per, key=lambda x: x['box'])
k = len(q) // 3
print(f"{'box size tercile':<34}{'median box':>12}{'full':>9}{'crop':>9}{'delta':>9}{'n':>5}")
for lab, grp in (('tightest third (real crop)', q[:k]),
                 ('middle third', q[k:2*k]),
                 ('loosest third (~whole frame)', q[2*k:])):
    print(f"  {lab:<32}{st.median(x['box'] for x in grp)*100:>11.1f}%"
          f"{st.mean(x['full'] for x in grp):>9.3f}{st.mean(x['crop'] for x in grp):>9.3f}"
          f"{st.mean(x['delta'] for x in grp):>+9.3f}{len(grp):>5}")

xs = [x['box'] for x in per]; ys = [x['delta'] for x in per]
n = len(xs); mx, my = st.mean(xs), st.mean(ys)
sx = math_sx = (sum((a - mx) ** 2 for a in xs)) ** .5
sy = (sum((b - my) ** 2 for b in ys)) ** .5
r = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)
t = r * ((n - 2) / (1 - r * r)) ** .5
print(f"\ncorrelation between box size and crop advantage: r = {r:+.3f}, t = {t:.2f} on {n-2} df")
print("  (negative r means the advantage grows as the box gets tighter)")

print(f"\n{'class':<28}{'full':>8}{'crop':>8}{'delta':>9}")
for lab in labels:
    g = [x for x in per if x['label'] == lab]
    print(f"  {lab:<26}{st.mean(x['full'] for x in g):>8.3f}"
          f"{st.mean(x['crop'] for x in g):>8.3f}{st.mean(x['delta'] for x in g):>+9.3f}")
json.dump(per, open('data/reports/species-compare-per-image.json', 'w'), indent=2)
