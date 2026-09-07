# Variable-tracking pilot execution

The original16-case run completed with exact reference/noop equality and every
reference score0. All signed losses were0, but this is not valid evidence about
VT competence: every reference spent the30-token budget on an answer introduction.
Preserve results/vt-pilot-v1 and results/vt-pilot-v1-summary.json unchanged.

Competing explanations were insufficient model competence, ordinary verbosity
under the fixed cap, or a missing supplied answer prefix. The cheapest check
resolved the third: official raw records contain answer_prefix separately from
input, and vendor/ruler/scripts/pred/call_api.py concatenates both before
generation. prepare_vt_pilot.py copied input but omitted answer_prefix. The
observed generated introductions reproduce the omitted prefix.

The correction supplies that exact prefix as native Qwen assistant prefill,
then allows the same30 new tokens. It does not regenerate contexts, change
answers, length, compression fraction, score, seed or viability thresholds.
The fixed B0 boundary is now at the end of the complete supplied input,
including its answer prefix. This changes the erroneous original boundary and
must not be presented as exact cache-state parity with the failed protocol.
The old manifest, raw data, metadata, runner source and outputs remain available.

Corrected first-case reproduction must pass before the full16 rerun. Until then
049 remains technically untested under its intended complete-input protocol.

The corrected first case passed on Orion: reference output contains all five
exact variable names, score1.0, with the same30-token cap and valid state/noop
checks. It uses4073 prompt tokens including the supplied prefix, native
continue_final_message mode, and exact prefix SHA. Full16 corrected rerun completed.
Runner SHA5cc05f71ce2b71ba55bdf930524157cd26300f985c6d5315b1d3c80279fe4e77.
Corrected manifest SHA596de1bcf78e2f3c4cb2c39b6483c5656cccaa8d99c5f80a550edabad3ebc9c1.

The corrected pilot produced16 perfect references and16 perfect compressed
answers, signed loss0 in every case. Reference competence passes, but all
variation/partial-loss/positive-loss gates fail. This fixedVT slice is not viable
for the stated prediction assay. Do not reinterpret the original prefix failure
as incompetence or the corrected zero effects as universal compression safety.
Independent final48-score/state/noop/prefix audit passed on all16 cases.
Report results/vt-pilot-independent-audit.json,
SHA1511e8d1a74082abce032def3f40712992e889ef20ebcbf69075214d94f75f3d.
Supplemental full-prefill versus split-prefill replay matched15/16; no such
parity is claimed. The frozen reference uses the shared boundary, whose paired
state and noop checks pass16/16. Preserve all outcomes. FixedVT slice closed.
See results/vt-pilot-v1-corrected-summary.json.
