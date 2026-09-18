"""Compact taxonomy of failure modes across 80 arms. Prints aggregates only."""
import json
from collections import Counter

TRUTH = {
    "113-competitor-v1-00": ("3705852", "3704852"),
    "113-competitor-v1-01": ("4860155", "4860455"),
    "113-competitor-v1-02": ("6109204", "6109704"),
    "113-competitor-v1-03": ("6024061", "6024064"),
}
base = "/tmp/herald-mech2/"
arms = []
for fn in ("dose.json", "pinned.json", "streaming.json", "excise.json"):
    for r in json.load(open(base + fn)):
        row = r["row"]
        pair = "-".join(row.split("-")[:4])
        truth, dist = TRUTH[pair]
        items = r["arms"].items() if "arms" in r else [("excise", r["arm"])]
        for name, a in items:
            arms.append((row, name, a, truth, dist))


def digits(s):
    return "".join(ch for ch in s if ch.isdigit())


def classify(a, truth, dist):
    if a["score"] == 1.0:
        return "preserved"
    runs = [x for x in a["runs"] if len(x) == 7]
    if not runs:
        return "collapse" if not a["runs"] else "truncation"
    if any(d == dist for d in runs):
        return "swap"
    if any(d == truth for d in runs):
        return "truncation"
    return "wrong-number"


print("== 1. first-divergence step vs reference path (approx: first step where top1 differs) ==")
print("(ref path unavailable here; use first digit-step position + carrier id instead)")
for mode in ("truncation", "swap", "wrong-number", "collapse"):
    sub = [(r, n, a) for r, n, a, t, d in arms if classify(a, t, d) == mode]
    digpos = []
    for r, n, a in sub:
        ds = [t for t, s in enumerate(a["steps"]) if 15 <= s["top"][0][0] <= 24]
        digpos.append(ds[0] if ds else None)
    print(f"  {mode:12s} n={len(sub):2d} first_digit_step={Counter(digpos)}")

print("== 2. wrong-number/swap edit structure (truth vs emitted runs) ==")
for r, n, a, t, d in arms:
    m = classify(a, t, d)
    if m in ("swap", "wrong-number"):
        for run in [x for x in a["runs"] if len(x) in (6, 7)]:
            TT = t[: len(run)]
            flips = [(i, x, y) for i, (x, y) in enumerate(zip(TT, run)) if x != y]
            dto = [i for i, (x, y) in enumerate(zip(t, d)) if x != y]
            print(f"  {r[20:]:9s} {n:10s} out={run} flips={flips} truth/dist-disagree={dto}")

print("== 3. collapse faces ==")
for r, n, a, t, d in arms:
    if classify(a, t, d) == "collapse":
        print(f"  {r[20:]:9s} {n:10s} len={a['len']} term={a['term']} nruns={len(a['runs'])}")

print("== 4. carrier step (~12) by mode ==")
for mode in ("preserved", "truncation", "swap", "wrong-number", "collapse"):
    sub = [a for r, n, a, t, d in arms if classify(a, t, d) == mode]
    ids = Counter(s["steps"][12]["top"][0][0] for s in sub if len(s["steps"]) > 12)
    ps = [s["steps"][12]["top"][0][1] for s in sub if len(s["steps"]) > 12]
    print(f"  {mode:12s} n={len(sub):2d} id12={dict(ids)} p_mean={sum(ps)/len(ps):.3f}" if ps else f"  {mode} empty")

print("== 5. length/term by mode ==")
for mode in ("preserved", "truncation", "swap", "wrong-number", "collapse"):
    sub = [a for r, n, a, t, d in arms if classify(a, t, d) == mode]
    print(f"  {mode:12s} n={len(sub):2d} lens={sorted(a['len'] for a in sub)} terms={Counter(a['term'] for a in sub)}")
