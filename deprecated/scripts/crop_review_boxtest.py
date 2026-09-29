"""Does 'uncertain' mean 'loosely cropped'? A permutation test on box_frac.

The reviewer's rationale mentioned that the uncertain images 'are not cropping
close to the plant'. reports/species-crop-comparison.md measured the opposite
sign for accuracy (looser boxes scored higher), so this checks whether the
uncertain set is actually loose-cropped at all.
"""
import json, pathlib, random, statistics as st

REVIEW = pathlib.Path(__file__).resolve().parents[1] / 'reports/crop-review'
rows = [json.loads(l) for l in (REVIEW / 'crop-review-manifest.jsonl').read_text().splitlines() if l.strip()]
fails = {int(x) for x in (REVIEW / 'failures.txt').read_text().split()}
unc = {int(x) for x in (REVIEW / 'uncertain.txt').read_text().split()}

def perm(a, b, n=20000, seed=20260921):
    obs = st.mean(a) - st.mean(b)
    pool = list(a) + list(b); k = len(a)
    rng = random.Random(seed); hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:k]) - st.mean(pool[k:])) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n + 1)

passes = [r['box_frac'] for r in rows if r['index'] not in fails | unc]
for label, sel in (('uncertain', unc), ('failures', fails)):
    grp = [r['box_frac'] for r in rows if r['index'] in sel]
    d, p = perm(grp, passes)
    print(f"{label:<10} n={len(grp):>3}  mean box {st.mean(grp)*100:5.1f}%  vs passes "
          f"{st.mean(passes)*100:5.1f}%   diff {d*100:+5.1f} pts   permutation p = {p:.3f}")
