# HERALD OpenAI Showcase Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deliver an 85-second, 1920 by 1080 HERALD product film for an OpenAI inference-team audience, using only artifact-backed research claims.

**Architecture:** A dedicated Remotion project renders seven modular scenes from a small generated evidence file. A Python extractor derives every displayed metric and the GSM8K example from current HERALD artifacts, while a second deterministic script creates the original score. Scene primitives remain independent from evidence extraction so claims can be revalidated without changing motion code.

**Tech Stack:** Remotion 4.0.487, React 19.2.7, TypeScript 7.0.2, Vitest 4.1.10, Python 3 through the repository's `uv` environment, Edge TTS through `uvx`, FFmpeg 8, H.264/AAC output.

## Global Constraints

- Render exactly 1920 by 1080 at 30 frames per second and 2550 frames, which is 85 seconds.
- Present HERALD to OpenAI's inference team without implying OpenAI adoption, endorsement, integration, or deployment.
- Use only current repository artifacts for experimental text and numbers.
- Call the main savings metric "compressed-generation fraction," never "memory savings."
- Keep the three live models compressor-specific and do not claim cross-compressor transfer.
- Do not claim zero quality loss, generalization beyond current Llama evidence, or total end-to-end wall overhead below 15%.
- The phrase "before it reaches the user" applies only to the private two-token probe and rollback path.
- Use warm alabaster, deep graphite, muted signal blue, and restrained coral. Exclude generic AI imagery, fake terminals, glowing brains, decorative bouncing, and excessive particles.
- Every important narrated claim must also be readable on screen.
- Keep generated MP4, WAV, AIFF, PNG frame renders, and installed dependencies out of Git.
- Preserve the user's existing `AGENTS.md` and `.Codex/` working-tree changes.

---

## File Map

- `showcase/herald-video/package.json`: pinned dependencies and render, still, test, and type-check commands.
- `showcase/herald-video/tsconfig.json`: strict TypeScript configuration.
- `showcase/herald-video/vitest.config.ts`: deterministic component and data tests.
- `showcase/herald-video/src/index.ts`: Remotion registration entrypoint.
- `showcase/herald-video/src/Root.tsx`: the 85-second composition definition.
- `showcase/herald-video/src/HeraldFilm.tsx`: scene sequencing, score, and narration.
- `showcase/herald-video/src/theme.ts`: colors, typography, spacing, easing, and safe-area constants.
- `showcase/herald-video/src/data/evidence.ts`: typed import of generated evidence JSON.
- `showcase/herald-video/src/data/evidence.generated.json`: generated, checked-in evidence used by the render.
- `showcase/herald-video/src/components/`: token stream, cache diagram, signal trace, metric lane, evidence label, and end-card primitives.
- `showcase/herald-video/src/scenes/`: seven storyboard scenes with no cross-scene state.
- `showcase/herald-video/src/__tests__/`: tests for timing, evidence loading, copy safety, and representative component renderability.
- `showcase/herald-video/scripts/build_evidence.py`: derives claims and text excerpts from HERALD artifacts.
- `showcase/herald-video/scripts/generate_score.py`: creates an original 85-second stereo WAV score.
- `showcase/herald-video/scripts/check_render.py`: validates final media metadata and audio presence.
- `showcase/herald-video/scripts/make_contact_sheet.sh`: renders representative frames and assembles a contact sheet.
- `showcase/herald-video/public/audio/narration.txt`: final narration script.
- `showcase/herald-video/public/audio/narration.mp3`: generated narration, not committed.
- `showcase/herald-video/public/audio/score.wav`: generated score, not committed.
- `showcase/herald-video/out/herald-openai-showcase.mp4`: final deliverable, not committed.
- `showcase/herald-video/out/contact-sheet.jpg`: representative-frame review artifact, not committed.

---

### Task 1: Scaffold the deterministic Remotion project

**Files:**
- Create: `showcase/herald-video/package.json`
- Create: `showcase/herald-video/tsconfig.json`
- Create: `showcase/herald-video/vitest.config.ts`
- Create: `showcase/herald-video/.gitignore`
- Create: `showcase/herald-video/src/index.ts`
- Create: `showcase/herald-video/src/Root.tsx`
- Test: `showcase/herald-video/src/__tests__/composition.test.tsx`

**Interfaces:**
- Consumes: no earlier task output.
- Produces: composition id `HeraldOpenAIShowcase`, constants `FPS`, `WIDTH`, `HEIGHT`, `DURATION_FRAMES`.

- [ ] **Step 1: Scaffold the blank project**

Run from the repository root:

```bash
mkdir -p showcase
npx create-video@4.0.487 --yes --blank --no-tailwind showcase/herald-video
```

Expected: `showcase/herald-video/` contains a working blank Remotion project.

- [ ] **Step 2: Replace the package manifest with pinned tooling**

Use this complete `package.json`:

```json
{
  "name": "herald-openai-showcase",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "studio": "remotion studio src/index.ts",
    "typecheck": "tsc --noEmit",
    "test": "vitest run",
    "still": "remotion still src/index.ts HeraldOpenAIShowcase",
    "render": "remotion render src/index.ts HeraldOpenAIShowcase out/herald-openai-showcase.mp4 --codec=h264 --crf=16 --audio-codec=aac --audio-bitrate=320k"
  },
  "dependencies": {
    "@fontsource/dm-serif-display": "5.2.8",
    "@fontsource/ibm-plex-mono": "5.2.7",
    "@fontsource/inter": "5.2.8",
    "@remotion/cli": "4.0.487",
    "@remotion/media": "4.0.487",
    "react": "19.2.7",
    "react-dom": "19.2.7",
    "remotion": "4.0.487",
    "zod": "4.4.3"
  },
  "devDependencies": {
    "@types/react": "19.2.17",
    "@types/react-dom": "19.2.3",
    "typescript": "7.0.2",
    "vitest": "4.1.10"
  }
}
```

- [ ] **Step 3: Add strict TypeScript and Vitest configuration**

Use `tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["DOM", "ES2022"],
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "jsx": "react-jsx",
    "resolveJsonModule": true,
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true,
    "types": ["vitest/globals"]
  },
  "include": ["src", "vitest.config.ts"]
}
```

Use `vitest.config.ts`:

```ts
import {defineConfig} from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
  },
});
```

- [ ] **Step 4: Write the failing composition contract test**

```tsx
import {describe, expect, it} from 'vitest';
import {DURATION_FRAMES, FPS, HEIGHT, WIDTH} from '../Root';

describe('HERALD film contract', () => {
  it('is an 85-second 1080p composition at 30fps', () => {
    expect({DURATION_FRAMES, FPS, WIDTH, HEIGHT}).toEqual({
      DURATION_FRAMES: 2550,
      FPS: 30,
      WIDTH: 1920,
      HEIGHT: 1080,
    });
  });
});
```

- [ ] **Step 5: Run the test and observe the expected failure**

Run:

```bash
cd showcase/herald-video
npm install
npm test
```

Expected: FAIL because the exported composition constants do not exist yet.

- [ ] **Step 6: Implement registration and composition constants**

Use `src/index.ts`:

```ts
import {registerRoot} from 'remotion';
import {RemotionRoot} from './Root';

registerRoot(RemotionRoot);
```

Use the initial `src/Root.tsx`:

```tsx
import {Composition} from 'remotion';
import {HeraldFilm} from './HeraldFilm';

export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;
export const DURATION_FRAMES = 2550;

export const RemotionRoot = () => (
  <Composition
    id="HeraldOpenAIShowcase"
    component={HeraldFilm}
    durationInFrames={DURATION_FRAMES}
    fps={FPS}
    width={WIDTH}
    height={HEIGHT}
  />
);
```

Create a temporary `src/HeraldFilm.tsx` that is valid until Task 4:

```tsx
import {AbsoluteFill} from 'remotion';

export const HeraldFilm = () => <AbsoluteFill style={{background: '#F3EFE6'}} />;
```

Use `.gitignore`:

```gitignore
node_modules/
out/
public/audio/*.mp3
public/audio/*.wav
public/audio/*.aiff
```

- [ ] **Step 7: Verify and commit the scaffold**

Run:

```bash
npm test
npm run typecheck
```

Expected: both commands PASS.

Commit only this task's files:

```bash
git add showcase/herald-video
git commit -m "scaffold HERALD showcase film"
```

---

### Task 2: Build the artifact-backed evidence pipeline

**Files:**
- Create: `showcase/herald-video/scripts/build_evidence.py`
- Create: `showcase/herald-video/scripts/test_build_evidence.py`
- Create: `showcase/herald-video/src/data/evidence.generated.json`
- Create: `showcase/herald-video/src/data/evidence.ts`
- Test: `showcase/herald-video/src/__tests__/evidence.test.ts`

**Interfaces:**
- Consumes: `results/live_controller_v3/{episodes,baseline}.jsonl`, `results/sweep/llama/gsm8k/references/gsm8k-0.json`, `results/sweep/llama/gsm8k/hybrids/streaming_llm__0.7500.jsonl`.
- Produces: `build_evidence(repo_root: Path) -> dict[str, Any]` and typed export `evidence: Evidence`.

- [ ] **Step 1: Write failing Python tests for the evidence contract**

```python
from pathlib import Path

from build_evidence import build_evidence


ROOT = Path(__file__).resolve().parents[3]


def test_live_campaign_metrics_match_current_artifacts() -> None:
    evidence = build_evidence(ROOT)
    assert evidence["campaign"]["episode_count"] == 552
    assert evidence["campaign"]["prompt_count"] == 46
    assert evidence["campaign"]["ratio_count"] == 4
    expected = {
        "expected_attention": (0.7937, 0.0027, 0.020),
        "knorm": (0.1208, 0.0118, 0.084),
        "streaming_llm": (0.3620, 0.0208, 0.051),
    }
    for compressor, values in expected.items():
        row = evidence["campaign"]["compressors"][compressor]
        assert round(row["compressed_generation_fraction"], 4) == values[0]
        assert round(row["quality_cost"], 4) == values[1]
        assert round(row["revert_wall_overhead"], 3) == values[2]


def test_gsm8k_example_is_the_real_s128_failure() -> None:
    example = build_evidence(ROOT)["gsm8k_example"]
    assert example["prompt_id"] == "gsm8k-0"
    assert example["switch_position"] == 128
    assert example["ratio"] == 0.75
    assert example["reference_answer"] == "18"
    assert example["compressed_answer"] == "3"
    assert example["compressed_quality"] == 0.0
```

- [ ] **Step 2: Run the Python tests and observe the expected failure**

Run:

```bash
uv run pytest showcase/herald-video/scripts/test_build_evidence.py -v
```

Expected: FAIL because `build_evidence.py` does not exist.

- [ ] **Step 3: Implement deterministic extraction**

Implement `build_evidence.py` with these exact public functions and formulas:

```python
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


COMPRESSORS = (
    "expected_attention",
    "knorm",
    "streaming_llm",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric")
    return sum(values) / len(values)


def compressor_metrics(
    episodes: list[dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
) -> dict[str, float | int]:
    savings: list[float] = []
    costs: list[float] = []
    revert_wall: list[float] = []
    token_overhead: list[float] = []
    for episode in episodes:
        baseline = baselines[episode["prompt_id"]]
        committed = episode["commit_s"] is not None
        savings.append(
            max(0.0, 1.0 - episode["commit_s"] / baseline["ref_len"])
            if committed
            else 0.0
        )
        costs.append(
            float(baseline["q_ref_live"]) - float(episode["q_live"])
            if committed
            else 0.0
        )
        reverted = [a for a in episode["attempts"] if not a["committed"]]
        revert_wall.append(
            sum(float(a["wall_s"]) for a in reverted) / float(baseline["wall_s"])
        )
        recorded_len = episode["ref_vs_recorded"]["recorded_len"]
        token_overhead.append(2.0 * len(reverted) / recorded_len)
    return {
        "episodes": len(episodes),
        "compressed_generation_fraction": mean(savings),
        "quality_cost": mean(costs),
        "revert_wall_overhead": mean(revert_wall),
        "token_overhead": mean(token_overhead),
    }


def build_evidence(repo_root: Path) -> dict[str, Any]:
    live_dir = repo_root / "results/live_controller_v3"
    all_episodes = load_jsonl(live_dir / "episodes.jsonl")
    baselines = {
        row["prompt_id"]: row
        for row in load_jsonl(live_dir / "baseline.jsonl")
    }
    by_compressor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in all_episodes:
        by_compressor[episode["compressor"]].append(episode)

    reference = json.loads(
        (repo_root / "results/sweep/llama/gsm8k/references/gsm8k-0.json").read_text()
    )
    hybrid_rows = load_jsonl(
        repo_root
        / "results/sweep/llama/gsm8k/hybrids/streaming_llm__0.7500.jsonl"
    )
    hybrid = next(
        row
        for row in hybrid_rows
        if row["prompt_id"] == "gsm8k-0" and row["s"] == 128
    )

    return {
        "campaign": {
            "episode_count": len(all_episodes),
            "prompt_count": len(baselines),
            "ratio_count": len({float(e["ratio"]) for e in all_episodes}),
            "compressor_count": len(COMPRESSORS),
            "compressors": {
                name: compressor_metrics(by_compressor[name], baselines)
                for name in COMPRESSORS
            },
        },
        "gsm8k_example": {
            "prompt_id": "gsm8k-0",
            "ratio": 0.75,
            "switch_position": 128,
            "reference_answer": reference["text"].rsplit("####", 1)[-1].strip(),
            "compressed_answer": hybrid["text"].rsplit("$", 1)[-1].split()[0],
            "reference_quality": reference["q"],
            "compressed_quality": hybrid["q"],
            "reference_excerpt": "9 eggs × $2 = $18",
            "compressed_excerpt": "6 eggs × $0.50 = $3",
        },
    }


def main() -> None:
    script = Path(__file__).resolve()
    repo_root = script.parents[3]
    output = repo_root / "showcase/herald-video/src/data/evidence.generated.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_evidence(repo_root), indent=2) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate JSON and add the TypeScript schema**

Run:

```bash
uv run python showcase/herald-video/scripts/build_evidence.py
```

Create `src/data/evidence.ts`:

```ts
import {z} from 'zod';
import raw from './evidence.generated.json';

const MetricRow = z.object({
  episodes: z.number().int().positive(),
  compressed_generation_fraction: z.number().min(0).max(1),
  quality_cost: z.number(),
  revert_wall_overhead: z.number().nonnegative(),
  token_overhead: z.number().nonnegative(),
});

const EvidenceSchema = z.object({
  campaign: z.object({
    episode_count: z.literal(552),
    prompt_count: z.literal(46),
    ratio_count: z.literal(4),
    compressor_count: z.literal(3),
    compressors: z.object({
      expected_attention: MetricRow,
      knorm: MetricRow,
      streaming_llm: MetricRow,
    }),
  }),
  gsm8k_example: z.object({
    prompt_id: z.literal('gsm8k-0'),
    ratio: z.literal(0.75),
    switch_position: z.literal(128),
    reference_answer: z.literal('18'),
    compressed_answer: z.literal('3'),
    reference_quality: z.literal(1),
    compressed_quality: z.literal(0),
    reference_excerpt: z.string(),
    compressed_excerpt: z.string(),
  }),
});

export type Evidence = z.infer<typeof EvidenceSchema>;
export const evidence: Evidence = EvidenceSchema.parse(raw);
```

- [ ] **Step 5: Add and run the TypeScript evidence test**

```ts
import {expect, it} from 'vitest';
import {evidence} from '../data/evidence';

it('loads only artifact-backed campaign evidence', () => {
  expect(evidence.campaign.episode_count).toBe(552);
  expect(evidence.gsm8k_example).toMatchObject({
    prompt_id: 'gsm8k-0',
    reference_answer: '18',
    compressed_answer: '3',
  });
});
```

Run:

```bash
uv run pytest showcase/herald-video/scripts/test_build_evidence.py -v
npm test
npm run typecheck
```

Expected: all commands PASS.

- [ ] **Step 6: Commit the evidence pipeline**

```bash
git add showcase/herald-video/scripts showcase/herald-video/src/data showcase/herald-video/src/__tests__/evidence.test.ts
git commit -m "add artifact-backed showcase evidence"
```

---

### Task 3: Create the visual foundation and reusable primitives

**Files:**
- Create: `showcase/herald-video/src/theme.ts`
- Create: `showcase/herald-video/src/components/Stage.tsx`
- Create: `showcase/herald-video/src/components/TokenStream.tsx`
- Create: `showcase/herald-video/src/components/CacheStack.tsx`
- Create: `showcase/herald-video/src/components/SignalTrace.tsx`
- Create: `showcase/herald-video/src/components/MetricLane.tsx`
- Create: `showcase/herald-video/src/components/EvidenceLabel.tsx`
- Test: `showcase/herald-video/src/__tests__/theme.test.ts`

**Interfaces:**
- Consumes: Remotion frame context and typed evidence rows.
- Produces: `Stage`, `TokenStream`, `CacheStack`, `SignalTrace`, `MetricLane`, `EvidenceLabel` components with typed props.

- [ ] **Step 1: Write the failing theme test**

```ts
import {expect, it} from 'vitest';
import {COLORS, SAFE_X, SAFE_Y} from '../theme';

it('uses the approved palette and 1080p safe area', () => {
  expect(COLORS).toEqual({
    alabaster: '#F3EFE6',
    graphite: '#171918',
    blue: '#5D78A6',
    coral: '#D9634F',
    sage: '#91A797',
    paper: '#FFFDF8',
  });
  expect({SAFE_X, SAFE_Y}).toEqual({SAFE_X: 120, SAFE_Y: 86});
});
```

- [ ] **Step 2: Implement the theme**

```ts
import '@fontsource/dm-serif-display/400.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/ibm-plex-mono/400.css';
import {Easing} from 'remotion';

export const COLORS = {
  alabaster: '#F3EFE6',
  graphite: '#171918',
  blue: '#5D78A6',
  coral: '#D9634F',
  sage: '#91A797',
  paper: '#FFFDF8',
} as const;

export const FONT = {
  display: 'DM Serif Display',
  sans: 'Inter',
  mono: 'IBM Plex Mono',
} as const;

export const SAFE_X = 120;
export const SAFE_Y = 86;
export const EASE = Easing.bezier(0.16, 1, 0.3, 1);
```

- [ ] **Step 3: Implement the primitive contracts**

Each primitive must be a pure component whose animation comes only from `useCurrentFrame()` and `interpolate()`. Use these prop interfaces exactly:

```ts
export type TokenStreamProps = {
  tokens: string[];
  state: 'stable' | 'unstable' | 'probe' | 'committed';
  revealFrame?: number;
};

export type CacheStackProps = {
  compression: number;
  heldInReserve: boolean;
  alarmed: boolean;
};

export type SignalTraceProps = {
  label: string;
  values: number[];
  accent: 'blue' | 'coral' | 'sage';
  progress: number;
};

export type MetricLaneProps = {
  name: string;
  fraction: number;
  qualityCost: number;
  overhead: number;
  delay: number;
};
```

`Stage` must establish the alabaster background, 120px horizontal safe area, 86px vertical safe area, and a subtle 1.5% paper-grain SVG overlay. `EvidenceLabel` must render the label `LIVE LLAMA · IFEVAL · ORION` and never exceed 20px.

- [ ] **Step 4: Run tests, type checking, and a primitive still**

Run:

```bash
npm test
npm run typecheck
npx remotion still src/index.ts HeraldOpenAIShowcase out/foundation.png --frame=0
```

Expected: tests and type check PASS; the still is 1920 by 1080 with no browser-console errors.

- [ ] **Step 5: Commit the visual foundation**

```bash
git add showcase/herald-video/src/theme.ts showcase/herald-video/src/components showcase/herald-video/src/__tests__/theme.test.ts
git commit -m "build HERALD film visual system"
```

---

### Task 4: Build the problem and real-failure scenes

**Files:**
- Create: `showcase/herald-video/src/scenes/HiddenCostScene.tsx`
- Create: `showcase/herald-video/src/scenes/FailureScene.tsx`
- Modify: `showcase/herald-video/src/HeraldFilm.tsx`
- Test: `showcase/herald-video/src/__tests__/copy-safety.test.ts`

**Interfaces:**
- Consumes: `TokenStream`, `CacheStack`, `Stage`, `evidence.gsm8k_example`.
- Produces: scene components sized to their local sequence duration.

- [ ] **Step 1: Write a failing copy-safety test**

```ts
import {expect, it} from 'vitest';
import {ON_SCREEN_COPY} from '../scenes/copy';

it('does not overclaim customer status or memory savings', () => {
  const all = ON_SCREEN_COPY.join(' ').toLowerCase();
  expect(all).not.toContain('openai uses');
  expect(all).not.toContain('deployed at openai');
  expect(all).not.toContain('memory savings');
  expect(all).not.toContain('zero quality loss');
});
```

- [ ] **Step 2: Create the centralized on-screen copy**

Create `src/scenes/copy.ts`:

```ts
export const ON_SCREEN_COPY = [
  'More context.',
  'Less memory.',
  'One hidden cost.',
  'Compression can corrupt an answer silently.',
  'The warning is already inside the model.',
  'No additional forward pass.',
  'Two private tokens. One safe decision.',
  'A safety layer for adaptive KV-cache compression.',
  'Compression, with an undo button.',
] as const;
```

- [ ] **Step 3: Implement the 0 to 7 second hidden-cost scene**

Use local frames 0 to 209. Reveal the first three statements one at a time, then let the cache shrink from 100% to 25%. The token stream must remain stable until local frame 145, then shift one token into the coral unstable state. Use an 18-frame ease for each reveal and no spring motion.

- [ ] **Step 4: Implement the 7 to 18 second failure scene**

Use local frames 0 to 329. Display two large answer cards from the evidence JSON. The left card resolves to `9 eggs × $2 = $18`. The right card begins from the same problem, crosses the switch marker at token 128, then resolves to `6 eggs × $0.50 = $3`. Label the right card `STREAMINGLLM · RATIO 0.75 · SWITCH 128` and the left card `UNCOMPRESSED REFERENCE`.

Do not render long generated paragraphs. Use the artifact-backed excerpts and a small source label.

Wire these two scenes into `HeraldFilm.tsx` at frames 0 and 210 so the task's representative stills exercise the real components. Keep the remaining timeline as the alabaster background until later scene tasks land.

- [ ] **Step 5: Verify scene stills and copy safety**

Run:

```bash
npm test
npm run typecheck
npx remotion still src/index.ts HeraldOpenAIShowcase out/frame-150.png --frame=150
npx remotion still src/index.ts HeraldOpenAIShowcase out/frame-390.png --frame=390
```

Expected: tests PASS; the first still shows the destabilizing token, and the second shows readable 18 versus 3 evidence cards.

- [ ] **Step 6: Commit the first scenes**

```bash
git add showcase/herald-video/src/scenes showcase/herald-video/src/__tests__/copy-safety.test.ts
git commit -m "animate HERALD problem and failure"
```

---

### Task 5: Build the signal and two-token mechanism scenes

**Files:**
- Create: `showcase/herald-video/src/scenes/SignalScene.tsx`
- Create: `showcase/herald-video/src/scenes/MechanismScene.tsx`
- Modify: `showcase/herald-video/src/HeraldFilm.tsx`
- Test: `showcase/herald-video/src/__tests__/timing.test.ts`

**Interfaces:**
- Consumes: `SignalTrace`, `TokenStream`, `CacheStack`, `Stage`.
- Produces: `SignalScene`, `MechanismScene`, and exported mechanism timing constants.

- [ ] **Step 1: Write a failing timing test**

```ts
import {expect, it} from 'vitest';
import {MECHANISM_TIMING} from '../scenes/MechanismScene';

it('keeps both probe tokens private before revert or commit', () => {
  expect(MECHANISM_TIMING.firstProbe).toEqual([180, 210]);
  expect(MECHANISM_TIMING.revert).toBe(224);
  expect(MECHANISM_TIMING.secondProbe).toEqual([330, 360]);
  expect(MECHANISM_TIMING.commit).toBe(374);
});
```

- [ ] **Step 2: Implement the 18 to 33 second signal scene**

Use local frames 0 to 449. Begin on the generated-token plane, then rotate the hierarchy into four signal rows: entropy, margin, KL change, rolling dynamics. Reveal the traces from left to right with deterministic SVG paths. The final 120 frames should connect the traces into a single risk indicator while displaying `NO ADDITIONAL FORWARD PASS`.

- [ ] **Step 3: Implement the 33 to 52 second mechanism scene**

Export these exact timing constants:

```ts
export const MECHANISM_TIMING = {
  firstProbe: [180, 210] as const,
  revert: 224,
  secondProbe: [330, 360] as const,
  commit: 374,
} as const;
```

Use local frames 0 to 569. The held cache remains visible behind the compressed cache. The first two probe tokens appear inside a clipped private lane, risk rises, a coral alarm fires, and the two tokens reverse out at frame 224. The second attempt repeats with sage state and commits at frame 374, after which the held cache fades away. The user-visible lane must never display the reverted tokens.

Wire the signal scene at global frame 540 and the mechanism scene at global frame 990 before rendering the critical stills.

- [ ] **Step 4: Verify critical mechanism frames**

Run:

```bash
npm test
npm run typecheck
npx remotion still src/index.ts HeraldOpenAIShowcase out/probe.png --frame=1170
npx remotion still src/index.ts HeraldOpenAIShowcase out/revert.png --frame=1195
npx remotion still src/index.ts HeraldOpenAIShowcase out/commit.png --frame=1345
```

Expected: the three stills show a private probe, a completed revert, and a safe commit with no reverted token in the output lane.

- [ ] **Step 5: Commit the mechanism**

```bash
git add showcase/herald-video/src/scenes/SignalScene.tsx showcase/herald-video/src/scenes/MechanismScene.tsx showcase/herald-video/src/__tests__/timing.test.ts
git commit -m "show HERALD two-token safety window"
```

---

### Task 6: Build the live-proof and OpenAI-framing scenes

**Files:**
- Create: `showcase/herald-video/src/scenes/ProofScene.tsx`
- Create: `showcase/herald-video/src/scenes/OpenAIFrameScene.tsx`
- Create: `showcase/herald-video/src/scenes/EndCardScene.tsx`
- Modify: `showcase/herald-video/src/HeraldFilm.tsx`
- Test: `showcase/herald-video/src/__tests__/proof.test.ts`

**Interfaces:**
- Consumes: `MetricLane`, `EvidenceLabel`, `evidence.campaign`.
- Produces: the final three scenes and `formatPercent(value: number, digits: number) -> string`.

- [ ] **Step 1: Write failing metric-format and claim tests**

```ts
import {expect, it} from 'vitest';
import {formatPercent, proofCopy} from '../scenes/ProofScene';

it('formats live metrics without renaming them as memory savings', () => {
  expect(formatPercent(0.7937, 1)).toBe('79.4%');
  expect(formatPercent(0.0027, 2)).toBe('0.27%');
  expect(proofCopy.toLowerCase()).toContain('compressed-generation fraction');
  expect(proofCopy.toLowerCase()).not.toContain('memory savings');
});
```

- [ ] **Step 2: Implement the 52 to 72 second proof scene**

Use local frames 0 to 599. Start with `552 LIVE HELD-OUT EPISODES`, then reveal three horizontal compressor lanes. Each lane shows compressed-generation fraction as the dominant bar, with quality cost and probe plus rollback overhead as smaller values. Use one decimal for the fraction and overhead, two decimals for quality cost percent.

Display:

```ts
export const proofCopy = 'Compressed-generation fraction · live Llama IFEval campaign';

export const formatPercent = (value: number, digits: number) =>
  `${(value * 100).toFixed(digits)}%`;
```

The accompanying line is `HERALD adapts its caution to the compressor.`

- [ ] **Step 3: Implement the 72 to 81 second OpenAI framing scene**

Use local frames 0 to 269. Draw a simple inference flow: `MODEL LOGITS` to `HERALD RISK` to `COMPRESSION CONTROLLER`. Add a separate `PRIVATE TWO-TOKEN WINDOW` loop into HERALD. The only OpenAI-specific text is the neutral audience label `FOR INFERENCE SYSTEMS TEAMS`. Do not use OpenAI's logo or imply integration.

- [ ] **Step 4: Implement the 81 to 85 second end card**

Use local frames 0 to 119. Hold the HERALD wordmark and `Compression, with an undo button.` for at least 90 frames. Use only alabaster, graphite, and a single blue cursor line.

Wire the proof scene at global frame 1560, the OpenAI framing scene at 2160, and the end card at 2430 before rendering their representative stills.

- [ ] **Step 5: Verify proof and end-card stills**

Run:

```bash
npm test
npm run typecheck
npx remotion still src/index.ts HeraldOpenAIShowcase out/proof.png --frame=1800
npx remotion still src/index.ts HeraldOpenAIShowcase out/openai-frame.png --frame=2250
npx remotion still src/index.ts HeraldOpenAIShowcase out/end-card.png --frame=2475
```

Expected: all numbers are readable at 100% scale, the OpenAI framing contains no customer claim, and the end card is visually stable.

- [ ] **Step 6: Commit the final scenes**

```bash
git add showcase/herald-video/src/scenes showcase/herald-video/src/__tests__/proof.test.ts
git commit -m "add HERALD live proof and pitch close"
```

---

### Task 7: Produce narration and original sound design

**Files:**
- Create: `showcase/herald-video/public/audio/narration.txt`
- Create: `showcase/herald-video/scripts/generate_score.py`
- Create: `showcase/herald-video/scripts/test_generate_score.py`
- Modify: `showcase/herald-video/src/HeraldFilm.tsx`

**Interfaces:**
- Consumes: approved story timings and scene sequence.
- Produces: `public/audio/narration.mp3`, `public/audio/score.wav`, and mixed audio in the composition.

- [ ] **Step 1: Write the final narration script**

Use this script exactly as the first timing pass:

```text
More context means a larger KV cache. Compression makes that cache cheaper, but the damage can arrive silently.

Here, the uncompressed model keeps the problem intact and answers eighteen. Under heavy compression, the reasoning changes. The model invents new facts and answers three.

HERALD looks for the warning inside the model's own next-token distribution: entropy, confidence margins, divergence, and rolling dynamics. Signals already produced during decoding. No additional forward pass.

When HERALD proposes compression, the full cache waits in reserve. Two compressed tokens are generated privately. If risk rises, those tokens are discarded and generation resumes from the safe cache. If the signal stays healthy, compression commits.

On Orion, the live controller completed five hundred fifty-two held-out IFEval episodes across three compressors and four ratios. It stayed compressed for seventy-nine percent of generation with ExpectedAttention, twelve percent with Knorm, and thirty-six percent with StreamingLLM, adapting its caution to each compressor. Probe and rollback overhead stayed between two and eight point four percent.

For inference systems teams, HERALD turns compression from a fixed gamble into an observable, reversible decision.

HERALD. Compression, with an undo button.
```

- [ ] **Step 2: Generate narration without repository credentials**

Run:

```bash
mkdir -p showcase/herald-video/public/audio
uvx --from edge-tts edge-tts \
  --voice en-US-AndrewMultilingualNeural \
  --rate=-5% \
  --file showcase/herald-video/public/audio/narration.txt \
  --write-media showcase/herald-video/public/audio/narration.mp3
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 showcase/herald-video/public/audio/narration.mp3
```

Expected: narration duration is between 74 and 82 seconds. If it falls outside that range, adjust only the `--rate` value and regenerate. Do not change research copy to solve timing.

- [ ] **Step 3: Write a failing score test**

```python
import wave
from pathlib import Path

from generate_score import OUTPUT, SAMPLE_RATE, generate_score


def test_score_is_85_second_stereo_without_clipping(tmp_path: Path) -> None:
    path = tmp_path / "score.wav"
    generate_score(path)
    with wave.open(str(path), "rb") as audio:
        assert audio.getnchannels() == 2
        assert audio.getframerate() == SAMPLE_RATE
        assert audio.getnframes() == SAMPLE_RATE * 85
    assert path.stat().st_size > 1_000_000
    assert OUTPUT.name == "score.wav"
```

- [ ] **Step 4: Implement the deterministic score**

Implement `generate_score.py` using NumPy from the repository environment. Generate stereo float samples at 48 kHz with these layers:

- a quiet D2/A2 pad with slow 8-second amplitude movement
- a low pulse at seconds 0, 18, 33, 52, 72, and 81
- subtle token ticks between seconds 33 and 52
- a descending filtered cue at 40.5 seconds for rollback
- an ascending resolved cue at 46.5 seconds for commit
- a 1.2-second fade in and 2.5-second fade out
- peak normalization to 0.72 before 16-bit PCM conversion

Expose:

```python
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 48_000
DURATION_SECONDS = 85
OUTPUT = Path(__file__).resolve().parents[1] / "public/audio/score.wav"


def generate_score(path: Path = OUTPUT) -> None:
    total = SAMPLE_RATE * DURATION_SECONDS
    time = np.arange(total, dtype=np.float64) / SAMPLE_RATE
    score = np.zeros(total, dtype=np.float64)

    slow_motion = 0.68 + 0.32 * np.sin(2 * np.pi * time / 8.0)
    score += slow_motion * (
        0.11 * np.sin(2 * np.pi * 73.42 * time)
        + 0.055 * np.sin(2 * np.pi * 110.0 * time)
    )

    for start in (0.0, 18.0, 33.0, 52.0, 72.0, 81.0):
        local = time - start
        active = (local >= 0) & (local < 2.4)
        score[active] += (
            0.20
            * np.sin(2 * np.pi * 55.0 * local[active])
            * np.exp(-2.0 * local[active])
        )

    for start in np.arange(33.25, 52.0, 0.5):
        local = time - start
        active = (local >= 0) & (local < 0.055)
        score[active] += (
            0.055
            * np.sin(2 * np.pi * 1250.0 * local[active])
            * np.exp(-72.0 * local[active])
        )

    rollback = time - 40.5
    active = (rollback >= 0) & (rollback < 0.9)
    x = rollback[active]
    score[active] += 0.13 * np.sin(
        2 * np.pi * (480.0 * x - 155.0 * x * x)
    ) * np.sin(np.pi * x / 0.9)

    commit = time - 46.5
    active = (commit >= 0) & (commit < 1.5)
    x = commit[active]
    envelope = np.sin(np.pi * x / 1.5) ** 2
    score[active] += envelope * (
        0.10 * np.sin(2 * np.pi * 261.63 * x)
        + 0.08 * np.sin(2 * np.pi * 392.0 * x)
    )

    fade_in = np.clip(time / 1.2, 0.0, 1.0)
    fade_out = np.clip((DURATION_SECONDS - time) / 2.5, 0.0, 1.0)
    score *= fade_in * fade_out
    score *= 0.72 / max(float(np.max(np.abs(score))), 1e-9)

    right = np.roll(score, int(0.011 * SAMPLE_RATE))
    right[: int(0.011 * SAMPLE_RATE)] = 0.0
    stereo = np.stack([score, right], axis=1)
    pcm = np.round(stereo * 32767.0).astype("<i2")

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm.tobytes())


if __name__ == "__main__":
    generate_score()
```

The implementation must use `numpy.sin`, deterministic envelopes, and `wave.open`; it must not use downloaded music or copyrighted samples.

- [ ] **Step 5: Generate and inspect audio**

Run:

```bash
uv run pytest showcase/herald-video/scripts/test_generate_score.py -v
uv run python showcase/herald-video/scripts/generate_score.py
ffmpeg -i showcase/herald-video/public/audio/score.wav -filter:a volumedetect -f null /dev/null
```

Expected: the test PASSes, duration is exactly 85.0 seconds, and `max_volume` is below 0 dB.

- [ ] **Step 6: Add audio to the film**

In `HeraldFilm.tsx`, render both tracks using `@remotion/media`:

```tsx
import {Audio} from '@remotion/media';
import {AbsoluteFill, Sequence, staticFile} from 'remotion';

export const HeraldFilm = () => (
  <AbsoluteFill>
    {/* Scene sequences from Task 8 */}
    <Audio src={staticFile('audio/score.wav')} volume={0.16} />
    <Sequence from={45}>
      <Audio src={staticFile('audio/narration.mp3')} volume={0.92} />
    </Sequence>
  </AbsoluteFill>
);
```

- [ ] **Step 7: Commit script and narration source**

```bash
git add showcase/herald-video/public/audio/narration.txt showcase/herald-video/scripts/generate_score.py showcase/herald-video/scripts/test_generate_score.py showcase/herald-video/src/HeraldFilm.tsx
git commit -m "add HERALD narration and original score"
```

---

### Task 8: Integrate scene timing and run draft visual review

**Files:**
- Modify: `showcase/herald-video/src/HeraldFilm.tsx`
- Create: `showcase/herald-video/src/scenes/timeline.ts`
- Test: `showcase/herald-video/src/__tests__/timeline.test.ts`

**Interfaces:**
- Consumes: all seven scene components and two audio files.
- Produces: final frame-accurate sequence with no gaps or overlaps.

- [ ] **Step 1: Write the failing timeline test**

```ts
import {expect, it} from 'vitest';
import {SCENES} from '../scenes/timeline';

it('covers all 2550 frames without gaps or overlaps', () => {
  expect(SCENES[0].from).toBe(0);
  for (let index = 1; index < SCENES.length; index += 1) {
    const previous = SCENES[index - 1];
    expect(SCENES[index].from).toBe(previous.from + previous.duration);
  }
  const last = SCENES.at(-1)!;
  expect(last.from + last.duration).toBe(2550);
});
```

- [ ] **Step 2: Implement the exact timeline**

```ts
export const SCENES = [
  {id: 'hidden-cost', from: 0, duration: 210},
  {id: 'failure', from: 210, duration: 330},
  {id: 'signal', from: 540, duration: 450},
  {id: 'mechanism', from: 990, duration: 570},
  {id: 'proof', from: 1560, duration: 600},
  {id: 'openai-frame', from: 2160, duration: 270},
  {id: 'end-card', from: 2430, duration: 120},
] as const;
```

- [ ] **Step 3: Wire all scenes with frame-local sequences**

Map each timeline item to a `<Sequence>` in `HeraldFilm.tsx`. Use `premountFor={30}` on each scene after the first to stabilize font and asset loading. Keep audio sequences after visual sequences so they remain independent of scene clipping.

- [ ] **Step 4: Run the complete automated check**

Run:

```bash
uv run pytest showcase/herald-video/scripts -v
cd showcase/herald-video
npm test
npm run typecheck
```

Expected: all Python tests, Vitest tests, and TypeScript checks PASS.

- [ ] **Step 5: Render a low-resolution draft**

Run:

```bash
npx remotion render src/index.ts HeraldOpenAIShowcase out/draft.mp4 --scale=0.5 --codec=h264 --crf=20
ffprobe -v error -show_entries format=duration -show_streams -of json out/draft.mp4
```

Expected: duration is 85 seconds, resolution is 960 by 540, frame rate is 30, and both video and audio streams exist.

- [ ] **Step 6: Review the draft from start to finish**

Watch the full draft. Record every issue in `showcase/herald-video/out/draft-review.md` using timestamp, problem, and concrete fix. Required review passes:

- story and claim accuracy
- 1080p-equivalent text readability
- visual hierarchy and safe areas
- transition continuity
- narration timing and pronunciation
- music and effect balance
- no frozen first or last frame at scene boundaries

Apply every concrete issue before continuing. Re-run tests after changes.

- [ ] **Step 7: Commit the integrated composition**

```bash
git add showcase/herald-video/src
git commit -m "integrate HERALD showcase timeline"
```

---

### Task 9: Render, inspect, and deliver the final film

**Files:**
- Create: `showcase/herald-video/scripts/check_render.py`
- Create: `showcase/herald-video/scripts/test_check_render.py`
- Create: `showcase/herald-video/scripts/make_contact_sheet.sh`
- Create: `showcase/herald-video/out/herald-openai-showcase.mp4`
- Create: `showcase/herald-video/out/contact-sheet.jpg`

**Interfaces:**
- Consumes: completed Remotion composition and audio assets.
- Produces: verified final MP4 and representative-frame contact sheet.

- [ ] **Step 1: Write the failing media-check test**

```python
from check_render import validate_probe


def test_final_probe_contract() -> None:
    validate_probe(
        {
            "format": {"duration": "85.000000"},
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080, "r_frame_rate": "30/1"},
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"},
            ],
        }
    )
```

- [ ] **Step 2: Implement strict final-media validation**

Implement `validate_probe(probe: dict[str, Any]) -> None` in `check_render.py`. Raise `ValueError` unless duration is between 84.95 and 85.05 seconds, the video is H.264 at 1920 by 1080 and 30 fps, and one AAC audio stream exists at 48 kHz. `main()` must invoke `ffprobe` with JSON output for `out/herald-openai-showcase.mp4` and print `PASS: final media contract` only after all assertions succeed.

- [ ] **Step 3: Render the full-resolution master**

Run:

```bash
cd showcase/herald-video
npm run render
```

Expected: `out/herald-openai-showcase.mp4` renders without dropped frames, browser-console errors, or missing assets.

- [ ] **Step 4: Validate the master**

Run:

```bash
uv run pytest scripts/test_check_render.py -v
uv run python scripts/check_render.py
```

Expected: both commands PASS.

- [ ] **Step 5: Build the contact sheet**

Create `make_contact_sheet.sh` to render frames 90, 360, 720, 1110, 1320, 1740, 2250, and 2475 into `out/frames/`, then call FFmpeg's `xstack` filter to produce a 4 by 2 `out/contact-sheet.jpg` at 1920 by 1080.

Run:

```bash
bash scripts/make_contact_sheet.sh
```

Expected: all eight stills exist and the contact sheet is 1920 by 1080.

- [ ] **Step 6: Perform final visual inspection**

Inspect every contact-sheet frame at full resolution. Then watch the master from start to finish with headphones. Confirm:

- evidence labels and footnotes are readable
- 18 versus 3 remains immediately understandable
- reverted probe tokens never enter the user-visible lane
- all three metric lanes use generated evidence values
- no OpenAI customer claim or logo appears
- narration is intelligible and music never masks it
- the end card holds cleanly through the final frame

If any item fails, fix the source, rerun automated checks, rerender the master, and repeat this inspection.

- [ ] **Step 7: Commit final source and validation helpers**

```bash
git add showcase/herald-video/scripts/check_render.py showcase/herald-video/scripts/test_check_render.py showcase/herald-video/scripts/make_contact_sheet.sh showcase/herald-video/package-lock.json
git commit -m "verify HERALD showcase master"
```

- [ ] **Step 8: Deliver absolute paths and verified metadata**

Report the final MP4, contact sheet, source project, duration, resolution, codec, and audio status. Do not claim completion until the master has been watched in full and `check_render.py` passes.
