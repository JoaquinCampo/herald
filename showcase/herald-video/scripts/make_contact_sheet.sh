#!/usr/bin/env bash

set -euo pipefail

frames=(90 360 720 1110 1320 1740 2250 2475)
mkdir -p out/frames

for frame in "${frames[@]}"; do
  npx remotion still src/index.ts HeraldOpenAIShowcase "out/frames/frame-$(printf '%06d' "$frame").png" --frame="$frame"
done

ffmpeg -y \
  -i out/frames/frame-000090.png \
  -i out/frames/frame-000360.png \
  -i out/frames/frame-000720.png \
  -i out/frames/frame-001110.png \
  -i out/frames/frame-001320.png \
  -i out/frames/frame-001740.png \
  -i out/frames/frame-002250.png \
  -i out/frames/frame-002475.png \
  -filter_complex "[0:v]scale=480:270,pad=480:540:0:135:color=#F3EFE6[a];[1:v]scale=480:270,pad=480:540:0:135:color=#F3EFE6[b];[2:v]scale=480:270,pad=480:540:0:135:color=#F3EFE6[c];[3:v]scale=480:270,pad=480:540:0:135:color=#F3EFE6[d];[4:v]scale=480:270,pad=480:540:0:135:color=#F3EFE6[e];[5:v]scale=480:270,pad=480:540:0:135:color=#F3EFE6[f];[6:v]scale=480:270,pad=480:540:0:135:color=#F3EFE6[g];[7:v]scale=480:270,pad=480:540:0:135:color=#F3EFE6[h];[a][b][c][d][e][f][g][h]xstack=inputs=8:layout=0_0|480_0|960_0|1440_0|0_540|480_540|960_540|1440_540" \
  -frames:v 1 -update 1 out/contact-sheet.jpg
