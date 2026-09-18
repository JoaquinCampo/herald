"""Behavioral atlas + signal inventory over 80 labeled arms (CPU-only).

Reads dose/pinned/streaming/excise JSONs. Prints:
  A. WHEN-map: per-row onset ratio and mode progression.
  B. Signal inventory: sensitivity/specificity/earliness per candidate.
Modes: preserved / truncation / swap / wrong-number / collapse.
Signals use only per-step top5/ent/mprob (online-computable quantities).
"""

import json

DIGITS = set(range(15, 25))
PROVIDED, TEXT, EOS = 3897, 1467, 151645
TRUTH = {
    "113-competitor-v1-00": ("3705852", "3704852"),
    "113-competitor-v1-01": ("4860155", "4860455"),
    "113-competitor-v1-02": ("6109204", "6109704"),
    "113-competitor-v1-03": ("6024061", "6024064"),
}
REF_LEN = 25


def load():
    arms = []  # (row, arm_name, payload, truth, distractor)
    base = "/tmp/herald-mech2/"
    for fn in ("dose.json", "pinned.json", "streaming.json", "excise.json"):
        for r in json.load(open(base + fn)):
            row = r["row"]
            pair = row.rsplit("-", 1)[0] if row[-1].isalpha() else row
            pair = "-".join(row.split("-")[:4])
            truth, dist = TRUTH[pair]
            if "arms" in r:
                for name, a in r["arms"].items():
                    arms.append((row, name, a, truth, dist))
            else:
                arms.append((row, "excise", r["arm"], truth, dist))
    return arms


def digits_of(runs):
    return [x for x in runs if len(x) == 7]


def classify(a, truth, dist):
    if a["score"] == 1.0:
        return "preserved"
    runs = digits_of(a["runs"])
    if not runs:
        return "collapse" if not a["runs"] else "truncation"
    if any(d == dist for d in runs):
        return "swap"
    if any(d == truth for d in runs):
        return "truncation"
    return "wrong-number"


def signals(a):
    steps = a["steps"]
    car = [(t, s) for t, s in enumerate(steps) if s["top"][0][0] in (PROVIDED, TEXT)]
    car_t, car_s = car[0] if car else (None, None)
    dig = [(t, s) for t, s in enumerate(steps) if s["top"][0][0] in DIGITS]
    early_ent = max([s["ent"] for t, s in enumerate(steps) if t < 16] + [0.0])
    dig_ent = max([s["ent"] for _, s in dig] + [0.0])
    min_margin = min([s["mprob"] for s in steps])
    runner = max([1.0 - s["top"][0][1] for _, s in dig] + [0.0])
    return {
        "carrier_t": car_t,
        "carrier_p": car_s["top"][0][1] if car_s else None,
        "early_ent": early_ent,
        "digit_ent": dig_ent,
        "min_margin": min_margin,
        "runnerup": runner,
        "len": a["len"],
    }


def main():
    arms = load()
    print("arms:", len(arms), "damaged:", sum(1 for _, _, a, _, _ in arms if a["score"] == 0.0))
    print("\n== A. WHEN-map (mode per arm) ==")
    rows = sorted(set(r for r, _, _, _, _ in arms))
    for row in rows:
        cells = []
        for r, name, a, truth, dist in arms:
            if r == row:
                cells.append("%s:%s" % (name.split(":")[-1], classify(a, truth, dist)[:5]))
        print(" ", row[20:], " ".join(cells))
    print("\n== B. signal inventory ==")
    lab = [(classify(a, t, d) != "preserved", signals(a), r, n) for r, n, a, t, d in arms]
    tests = [
        ("carrier_p<0.7", lambda s: (s["carrier_p"] or 1.0) < 0.7),
        ("early_ent>0.05", lambda s: s["early_ent"] > 0.05),
        ("digit_ent>0.02", lambda s: s["digit_ent"] > 0.02),
        ("digit_ent>0.10", lambda s: s["digit_ent"] > 0.10),
        ("min_margin<0.99", lambda s: s["min_margin"] < 0.99),
        ("runnerup>0.005", lambda s: s["runnerup"] > 0.005),
        ("len<25", lambda s: s["len"] < REF_LEN),
    ]
    for tname, fn in tests:
        tp = sum(1 for dmg, s, _, _ in lab if dmg and fn(s))
        fn_ = sum(1 for dmg, s, _, _ in lab if not dmg and fn(s))
        nd = sum(1 for dmg, _, _, _ in lab if dmg)
        np_ = sum(1 for dmg, _, _, _ in lab if not dmg)
        print("  %-16s sens=%d/%d spec=%d/%d" % (tname, tp, nd, np_ - fn_, np_))
    print("\n== C. earliness: first step with digit_ent contribution ==")
    for dmg, s, r, n in lab:
        if dmg:
            ds = [t for t, st in enumerate(next(a for rr, nn, a, _, _ in arms if rr == r and nn == n)["steps"])
                  if st["top"][0][0] in DIGITS and st["ent"] > 0.02]
            print("  %-28s %-12s first_soft_digit_step=%s carrier_t=%s carrier_p=%s len=%d" % (
                r[20:], n, ds[:3], s["carrier_t"], s["carrier_p"], s["len"]))


if __name__ == "__main__":
    main()
