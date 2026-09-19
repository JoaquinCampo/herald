"""Independent stdlib cross-check of mediation packet; no model imports.
Usage: python mediation_core.py HERALD_MEDIATION_TEXT_PACKET.md
This does not replace the full audit and staged runner in the companion bundle.
"""
import csv
import hashlib
import io
import json
import re
import struct
import sys
from pathlib import Path

EXPECTED = "e57e53661e9e35196ca59cc669d2b9943959efddde2ac5e3d86e0d38831a7be7"

def f32(x):
    return struct.unpack("f", struct.pack("f", float(x)))[0]

def table(text, heading):
    matches = re.findall(r"^````csv\n(.*?)^````", text, flags=re.M | re.S)
    blocks = [b for b in matches if b.startswith(heading)]
    if len(blocks) != 1:
        raise ValueError("CSV section not unique: " + heading)
    return list(csv.DictReader(io.StringIO(blocks[0])))

def run(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise ValueError("Unexpected packet hash")
    text = data.decode()
    decisions = table(text, "arm,offset,t,event,lead,")
    projections = table(text, "arm,offset,condition,candidate_ids,")
    continuations = table(text, "arm,offset,condition,recipient,token_ids,")
    if (len(decisions), len(projections), len(continuations)) != (450, 390, 75):
        raise AssertionError("Unexpected evidence coverage")
    key = lambda r: (r["arm"], int(r["offset"]), r["condition"])
    di = {key(r): r for r in decisions}
    pi = {key(r): r for r in projections}
    if len(di) != 450 or len(pi) != 390:
        raise AssertionError("Duplicate keys")
    out = []
    for r in decisions:
        if r["condition"] != "baseline_compressed" or r["lead"] != "0":
            continue
        arm, off = r["arm"], int(r["offset"])
        y, rival = int(r["target_id"]), str(r["strongest_nontarget_id"])
        def margin(condition):
            z = json.loads(di[(arm, off, condition)]["candidate_logits"])
            return f32(z[str(y)]) - f32(z[rival])
        def raw_sum(condition):
            pr = pi[(arm, off, condition)]
            j = json.loads(pr["candidate_ids"]).index(int(rival))
            return sum(f32(json.loads(pr[l])[j]) for l in ["MLP25", "MLP26", "MLP27"])
        c, h, frozen = map(margin, ["baseline_compressed", "ref_both_heads",
                                   "ref_heads_frozen_comp_MLPs"])
        denom = raw_sum("ref_late_MLPs") - raw_sum("self_mlp")
        out.append(dict(arm=arm, offset=off, baseline=c, readers=h, frozen=frozen,
            frozen_benefit=frozen-c, responsive_benefit=h-frozen,
            head_interaction=h-margin("ref_L22G0")-margin("ref_L23G2")+c,
            projected_gap_recovery=(raw_sum("ref_both_heads")-raw_sum("self_mlp"))/denom))
    if len(out) != 5:
        raise AssertionError("Expected five first errors")
    ci = {key(r): r for r in continuations}
    first = [r for r in continuations if r["first_number_exact"] == "true"]
    equal = sum(json.loads(r["token_ids"]) ==
                json.loads(ci[(r["arm"], int(r["offset"]), "baseline_reference")]["token_ids"])
                for r in first)
    return dict(scope="Exported-text cross-check, no GPU run", packet_sha256=EXPECTED,
        decisions=450, projections=390, continuations=75, first_errors=out,
        first_correct=len(first),
        any_exact=sum(float(r["exact_truth_score"]) for r in continuations),
        correct_full_ids_equal_reference=equal)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(run(sys.argv[1]), indent=2, sort_keys=True, allow_nan=False))
