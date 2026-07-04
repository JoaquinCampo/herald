"""Verify the persisted reference feature arrays are sound.

Loads stored .npy feature arrays from a results tree and checks shape,
finiteness, and value ranges against the documented feature semantics.
This validates the actual dataset on disk, not just the computation.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from herald.features import FEATURE_NAMES

IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir", type=Path, help="references directory")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    npys = sorted(glob.glob(str(args.ref_dir / "*.npy")))[: args.n]
    print(f"checking {len(npys)} feature arrays, {len(FEATURE_NAMES)} cols")
    ok = True
    for f in npys:
        a = np.load(f).astype(np.float64)
        jf = f[:-4] + ".json"
        gen_len = len(json.load(open(jf))["gen_ids"])
        problems = []
        if a.shape != (gen_len, len(FEATURE_NAMES)):
            problems.append(f"shape {a.shape} != ({gen_len}, n_feat)")
        # kl_prev[0] is NaN by design; everything else must be finite.
        kl = IDX["kl_prev"]
        mask = a.copy()
        mask[0, kl] = 0.0  # blank the one allowed NaN
        if not np.isfinite(mask).all():
            problems.append("non-finite entry outside kl_prev[0]")
        if a.shape[0] > 1 and not np.isnan(a[0, kl]):
            problems.append("kl_prev[0] should be NaN")
        if (a[:, IDX["entropy"]] < -1e-4).any():
            problems.append("negative entropy")
        if (a[:, IDX["varentropy"]] < -1e-4).any():
            problems.append("negative varentropy")
        # h_alts is the entropy of the renormalized non-top-1 mass; it is
        # >= 0 but can EXCEED total entropy (confident top, diffuse rest).
        if (a[:, IDX["h_alts"]] < -1e-4).any():
            problems.append("negative h_alts")
        if (a[:, IDX["logit_std"]] < -1e-4).any():
            problems.append("negative logit_std")
        if (a[:, IDX["logit_range"]] < -1e-4).any():
            problems.append("negative logit_range")
        if (a[:, IDX["avg_logp"]] > 1e-4).any():
            problems.append("avg_logp > 0 (impossible for a log-prob mean)")
        mp = a[:, IDX["max_prob"]]
        if (mp < -1e-6).any() or (mp > 1 + 1e-6).any():
            problems.append("max_prob out of [0,1]")
        for k in (2, 10, 100):
            col = a[:, IDX[f"topk_mass_{k}"]]
            if (col < -1e-6).any() or (col > 1 + 1e-6).any():
                problems.append(f"topk_mass_{k} out of [0,1]")
        klcol = a[1:, kl]
        if klcol.size and (klcol < -1e-4).any():
            problems.append("negative kl_prev")
        tag = "OK" if not problems else "FAIL " + "; ".join(problems)
        e = a[:, IDX["entropy"]]
        print(
            f"  {Path(f).name}: {a.shape} entropy[{e.min():.2f},"
            f"{e.max():.2f}] maxp[{mp.min():.2f},{mp.max():.2f}] {tag}"
        )
        ok = ok and not problems
    print("FEATURES_OK" if ok else "FEATURES_PROBLEM")


if __name__ == "__main__":
    main()
