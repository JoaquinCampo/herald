"""Aggregate Phase-J mass_all.jsonl into generality verdicts. Aggregates only."""
import json
import sys

recs = [json.loads(line) for line in open(sys.argv[1]).read().splitlines()]
print("records:", len(recs), "errors:", sum(1 for r in recs if "error" in r),
      "repro_fail:", sum(1 for r in recs if "error" not in r and not (r["repro_ref"] and r["repro_comp"])))
for r in recs:
    if "error" in r:
        print("ERR", r["mid"], r["arm"], r["t"], r["error"][:100])
    else:
        L22 = r["layers"]["22"]
        print("%s %s t=%d cosL22=%.4f needle=%.3f/%.3f disc_n=%.3f/%.3f disc_d=%.3f/%.3f last2=%.3f/%.3f" % (
            r["mid"][20:], r["arm"], r["t"], r["cos"][22],
            L22["ref"]["needle"], L22["comp"]["needle"],
            L22["ref"]["disc_n"], L22["comp"]["disc_n"],
            L22["ref"]["disc_d"], L22["comp"]["disc_d"],
            L22["ref"]["last2"], L22["comp"]["last2"]))
