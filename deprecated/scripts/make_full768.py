"""Full frames downsampled exactly as the crops were, to isolate the confound.

render_crop_review.py saved each crop via thumbnail((768,768), LANCZOS). The full
frames kept their original size, up to 1600px. Both arms then resize to 224, so
the crop arm received a two-stage downsample the full arm did not. This builds a
third arm: uncropped frames, same thumbnail call, so 'crop vs full768' isolates
cropping and 'full768 vs full' isolates resampling.
"""
import json, pathlib
from PIL import Image

SET = pathlib.Path('/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-species-set')
out = SET / 'full768'
out.mkdir(exist_ok=True)
rows = [json.loads(l) for l in open(SET / 'species-records.jsonl') if l.strip()]
for i, r in enumerate(rows, 1):
    with Image.open(r['image']) as im:
        im = im.convert('RGB')
        im.thumbnail((768, 768), Image.Resampling.LANCZOS)
        im.save(out / f"inat_{r['photo_id']}.jpg", quality=90)
    if i % 400 == 0:
        print(f"  {i}/{len(rows)}", flush=True)
print(f"wrote {len(list(out.glob('*.jpg')))} full-frame 768px images")
