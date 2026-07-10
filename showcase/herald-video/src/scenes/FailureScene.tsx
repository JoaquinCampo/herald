import type {ReactNode} from 'react';
import {interpolate, useCurrentFrame} from 'remotion';
import {Stage} from '../components/Stage';
import {evidence} from '../data/evidence';
import {COLORS, EASE, FONT} from '../theme';
import {ON_SCREEN_COPY} from './copy';

type AnswerCardProps = {
  accent: string;
  answer: string;
  excerpt: string;
  label: string;
  meta: ReactNode;
  progress: number;
};

const AnswerCard = ({accent, answer, excerpt, label, meta, progress}: AnswerCardProps) => (
  <div
    style={{
      display: 'flex',
      minWidth: 0,
      flex: 1,
      flexDirection: 'column',
      justifyContent: 'space-between',
      minHeight: 600,
      padding: '40px 44px 38px',
      borderTop: `5px solid ${accent}`,
      backgroundColor: COLORS.paper,
      boxShadow: '0 20px 60px rgba(23, 25, 24, 0.08)',
      opacity: progress,
      translate: `0 ${interpolate(progress, [0, 1], [30, 0])}px`,
    }}
  >
    <div style={{display: 'flex', flexDirection: 'column', gap: 18}}>
      <div
        style={{
          color: accent,
          fontFamily: FONT.mono,
          fontSize: 20,
          lineHeight: 1.3,
          letterSpacing: '0.065em',
        }}
      >
        {label}
      </div>
      <div style={{fontFamily: FONT.sans, fontSize: 27, lineHeight: 1.35, color: `${COLORS.graphite}A8`}}>
        {meta}
      </div>
    </div>

    <div style={{display: 'flex', flexDirection: 'column', gap: 22}}>
      <div
        style={{
          maxWidth: 650,
          fontFamily: FONT.mono,
          fontSize: 35,
          lineHeight: 1.42,
          letterSpacing: '-0.025em',
        }}
      >
        {excerpt}
      </div>
      <div style={{display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 20}}>
        <span style={{fontFamily: FONT.sans, fontSize: 24, letterSpacing: '0.02em'}}>FINAL ANSWER</span>
        <span style={{color: accent, fontFamily: FONT.display, fontSize: 126, lineHeight: 0.8}}>{answer}</span>
      </div>
    </div>
  </div>
);

export const FailureScene = () => {
  const frame = useCurrentFrame();
  const leftProgress = interpolate(frame, [12, 34], [0, 1], {
    easing: EASE,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const rightProgress = interpolate(frame, [82, 108], [0, 1], {
    easing: EASE,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const switchProgress = interpolate(frame, [58, 78], [0, 1], {
    easing: EASE,
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const {gsm8k_example: example} = evidence;

  return (
    <Stage style={{flexDirection: 'column', gap: 32}}>
      <div style={{display: 'flex', alignItems: 'end', justifyContent: 'space-between', gap: 48}}>
        <h1
          style={{
            maxWidth: 1120,
            margin: 0,
            fontFamily: FONT.display,
            fontSize: 72,
            fontWeight: 400,
            lineHeight: 1.02,
            letterSpacing: '-0.03em',
          }}
        >
          {ON_SCREEN_COPY[3]}
        </h1>
        <div
          style={{
            paddingBottom: 7,
            fontFamily: FONT.mono,
            fontSize: 19,
            letterSpacing: '0.08em',
            whiteSpace: 'nowrap',
          }}
        >
          ARTIFACT EXCERPT · {example.prompt_id.toUpperCase()}
        </div>
      </div>

      <div style={{display: 'flex', minHeight: 0, flex: 1, gap: 34}}>
        <AnswerCard
          accent={COLORS.blue}
          answer={example.reference_answer}
          excerpt={example.reference_excerpt}
          label="UNCOMPRESSED REFERENCE"
          meta="Same prompt · problem state preserved"
          progress={leftProgress}
        />
        <AnswerCard
          accent={COLORS.coral}
          answer={example.compressed_answer}
          excerpt={example.compressed_excerpt}
          label={`STREAMINGLLM · RATIO ${example.ratio.toFixed(2)} · SWITCH ${example.switch_position}`}
          meta={
            <span style={{display: 'flex', alignItems: 'center', gap: 13}}>
              <span
                style={{
                  display: 'inline-block',
                  width: 38,
                  height: 2,
                  background: COLORS.coral,
                  opacity: switchProgress,
                  scale: `${switchProgress} 1`,
                  transformOrigin: 'left center',
                }}
              />
              Same prompt · unrelated box state introduced
            </span>
          }
          progress={rightProgress}
        />
      </div>
    </Stage>
  );
};
