import {interpolate, useCurrentFrame} from 'remotion';
import {Stage} from '../components/Stage';
import {COLORS, EASE, FONT} from '../theme';

const clamp = {
  easing: EASE,
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const FlowNode = ({label, delay, color = COLORS.graphite}: {label: string; delay: number; color?: string}) => {
  const frame = useCurrentFrame();
  const reveal = interpolate(frame, [delay, delay + 22], [0, 1], clamp);
  return (
    <div
      style={{
        display: 'flex',
        minWidth: 280,
        alignItems: 'center',
        justifyContent: 'center',
        padding: '31px 26px',
        border: `2px solid ${color}`,
        color,
        fontFamily: FONT.mono,
        fontSize: 23,
        letterSpacing: '0.08em',
        opacity: reveal,
        translate: `0 ${interpolate(reveal, [0, 1], [20, 0])}px`,
      }}
    >
      {label}
    </div>
  );
};

const Arrow = ({delay}: {delay: number}) => {
  const frame = useCurrentFrame();
  const progress = interpolate(frame, [delay, delay + 18], [0, 1], clamp);
  return (
    <div style={{display: 'flex', alignItems: 'center', opacity: progress}}>
      <div style={{width: 88 * progress, height: 2, background: COLORS.blue}} />
      <div
        style={{
          width: 0,
          height: 0,
          borderTop: '7px solid transparent',
          borderBottom: '7px solid transparent',
          borderLeft: `11px solid ${COLORS.blue}`,
        }}
      />
    </div>
  );
};

export const OpenAIFrameScene = () => {
  const frame = useCurrentFrame();
  const loop = interpolate(frame, [70, 90], [0, 1], clamp);

  return (
    <Stage style={{flexDirection: 'column', justifyContent: 'space-between'}}>
      <div style={{display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between'}}>
        <div style={{display: 'flex', maxWidth: 980, flexDirection: 'column', gap: 13}}>
          <span style={{color: COLORS.blue, fontFamily: FONT.mono, fontSize: 20, letterSpacing: '0.12em'}}>
            INFERENCE CONTROL PLANE
          </span>
          <h1
            style={{
              margin: 0,
              fontFamily: FONT.display,
              fontSize: 86,
              fontWeight: 400,
              letterSpacing: '-0.04em',
              lineHeight: 0.98,
            }}
          >
            A safety layer for adaptive KV-cache compression.
          </h1>
        </div>
        <span style={{paddingTop: 20, fontFamily: FONT.mono, fontSize: 20, letterSpacing: '0.1em'}}>
          FOR INFERENCE SYSTEMS TEAMS
        </span>
      </div>

      <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 22}}>
        <FlowNode label="MODEL LOGITS" delay={18} color={COLORS.graphite} />
        <Arrow delay={34} />
        <FlowNode label="HERALD RISK" delay={46} color={COLORS.coral} />
        <Arrow delay={62} />
        <FlowNode label="COMPRESSION CONTROLLER" delay={74} color={COLORS.blue} />
      </div>

      <div
        style={{
          alignSelf: 'center',
          display: 'flex',
          alignItems: 'center',
          gap: 18,
          padding: '21px 30px',
          border: `1px solid ${COLORS.sage}`,
          color: COLORS.sage,
          fontFamily: FONT.mono,
          fontSize: 20,
          letterSpacing: '0.1em',
          opacity: loop,
          translate: `0 ${interpolate(loop, [0, 1], [18, 0])}px`,
        }}
      >
        <span style={{width: 11, height: 11, borderRadius: 11, background: COLORS.sage}} />
        PRIVATE TWO-TOKEN WINDOW
        <span style={{width: 66, height: 1, background: COLORS.sage}} />
        HERALD RISK
      </div>
    </Stage>
  );
};
