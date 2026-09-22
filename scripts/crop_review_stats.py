"""Crop-review failure rate, recomputed on the completed review.

The first pass was scored against an 11-line failures.txt returned before the
review was finished. The completed review adds index 191 and returns 17
`uncertain` indices, which the first pass had assumed did not exist.

The two labels answer different questions and are kept apart here:
  failures  -- the sundew is not inside the crop (crop containment)
  uncertain -- the reviewer could not confidently identify the species from the
               photograph: blur, poor quality, obscuring vegetation
Only the first is a segmentation failure. The second is an input-quality
property of the photograph that a better segmenter cannot fix.
"""
import json, math, pathlib, sys, statistics as st

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from sundew_segmentation.growth_forms import growth_form_for_taxon

# The review lists and the manifest are committed; the images and their
# licence metadata are not, and stay on the cluster.
REVIEW = REPO / 'reports/crop-review'
BASE = pathlib.Path('/cluster/VAST/mendozacozatld-lab/PierceTaylor/sundew-crop-review')
rows = [json.loads(l) for l in (REVIEW / 'crop-review-manifest.jsonl').read_text().splitlines() if l.strip()]
meta = {}
for l in (BASE / 'metadata.jsonl').read_text(encoding='utf-8').splitlines():
    if l.strip():
        r = json.loads(l)
        meta[pathlib.Path(r['local_path']).name] = r
for r in rows:
    m = meta.get(r['name'], {})
    r['species'] = m.get('taxon_name', 'unknown')
    r['growth_form'] = growth_form_for_taxon(r['species'])
    r['observer_seen'] = bool(m.get('observer_seen_in_labelled', False))

fails = {int(x) for x in (REVIEW / 'failures.txt').read_text().split()}
unc = {int(x) for x in (REVIEW / 'uncertain.txt').read_text().split()}
assert not (fails & unc), 'an index cannot be both'
by_index = {r['index']: r for r in rows}
missing = (fails | unc) - set(by_index)
assert not missing, f'indices not in manifest: {sorted(missing)}'
N = len(rows)


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def line(label, k, n):
    lo, hi = wilson(k, n)
    return f"{label:<46}{k:>4}/{n:<5}{k/n*100:>7.2f}%   [{lo*100:5.2f}%, {hi*100:5.2f}%]"


print("=" * 88)
print(f"CROP CONTAINMENT FAILURE RATE  (completed review, {N} images)")
print("=" * 88)
print(line("failures only (headline)", len(fails), N))
print(line("uncertain excluded from the denominator", len(fails), N - len(unc)))
print(line("uncertain counted as failures (upper bound)", len(fails | unc), N))
print()
print(line("[superseded] the 11-failure first pass", 11, N))
print()
print("Separately, as a corpus property rather than a model failure:")
print(line("photographs the reviewer could not ID", len(unc), N))

print("\n" + "=" * 88)
print("THE TWELVE FAILURES")
print("=" * 88)
print(f"{'idx':>5}  {'species':<26}{'growth form':<20}{'box':>6}{'conf':>8}   {'':<3}")
for i in sorted(fails):
    r = by_index[i]
    new = ' NEW' if i == 191 else ''
    print(f"{i:>5}  {r['species']:<26}{r['growth_form']:<20}{r['box_frac']*100:>5.0f}%{r['mean_conf_fg']:>8.3f}{new}")

print("\n" + "=" * 88)
print("THE SEVENTEEN UNCERTAIN (photo quality, not crop containment)")
print("=" * 88)
for i in sorted(unc):
    r = by_index[i]
    print(f"{i:>5}  {r['species']:<26}{r['growth_form']:<20}{r['box_frac']*100:>5.0f}%{r['mean_conf_fg']:>8.3f}")

print("\n" + "=" * 88)
print("STRATIFICATION BY GROWTH FORM  (failures only)")
print("=" * 88)
forms = {}
for r in rows:
    forms.setdefault(r['growth_form'], []).append(r['index'])
for form, idxs in sorted(forms.items(), key=lambda kv: -len(kv[1])):
    k = len(set(idxs) & fails)
    print(line(form, k, len(idxs)))

print("\nSame table with uncertain folded in, for contrast:")
for form, idxs in sorted(forms.items(), key=lambda kv: -len(kv[1])):
    k = len(set(idxs) & (fails | unc))
    print(line(form, k, len(idxs)))

print("\n" + "=" * 88)
print("OBSERVER FAMILIARITY")
print("=" * 88)
for label, want in (("observer seen in labelled set", True), ("novel observer", False)):
    idxs = {r['index'] for r in rows if r['observer_seen'] is want}
    print(line(label, len(idxs & fails), len(idxs)))

print("\n" + "=" * 88)
print("IS LOOSE CROPPING THE PROBLEM?  box_frac of failures vs uncertain vs passes")
print("=" * 88)
groups = {
    'failures': [by_index[i] for i in fails],
    'uncertain': [by_index[i] for i in unc],
    'passes': [r for r in rows if r['index'] not in fails | unc],
}
for name, g in groups.items():
    bf = [r['box_frac'] for r in g]
    cf = [r['mean_conf_fg'] for r in g]
    print(f"  {name:<12} n={len(g):>4}  median box {st.median(bf)*100:5.1f}%  "
          f"mean box {st.mean(bf)*100:5.1f}%  mean conf {st.mean(cf):.3f}")

print("\n" + "=" * 88)
print("CONFIDENCE AS A FILTER, recomputed on twelve")
print("=" * 88)
for t in (0.85, 0.90):
    flagged = [r for r in rows if r['mean_conf_fg'] < t]
    caught = len({r['index'] for r in flagged} & fails)
    prec = caught / len(flagged) if flagged else 0.0
    print(f"  conf < {t:.2f}   flags {len(flagged):>3} images ({len(flagged)/N*100:4.1f}%)   "
          f"catches {caught:>2} of {len(fails)} failures   precision {prec:.2f}")

print("\n" + "=" * 88)
print("AT SCALE, per 10,000 scraped images")
print("=" * 88)
for label, k, n in (("bad crops (failures)", len(fails), N),
                    ("unidentifiable photographs (uncertain)", len(unc), N),
                    ("either", len(fails | unc), N)):
    lo, hi = wilson(k, n)
    print(f"  {label:<40} ~{int(k/n*10000):>5}   ({int(lo*10000)}-{int(hi*10000)})")
