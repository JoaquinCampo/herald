import {interpolate, useCurrentFrame} from 'remotion';
import {CacheStack} from '../components/CacheStack';
import {Stage} from '../components/Stage';
import {TokenStream} from '../components/TokenStream';
import {COLORS, EASE, FONT} from '../theme';

export const MECHANISM_TIMING = {
  firstProbe: [180, 210] as const,
  revert: 224,
  secondProbe: [330, 360] as const,
  commit: 374,
} as const;

const BASE_TOKENS = ['The', 'answer', 'is'] as const;

export type MechanismState = {
  visibleTokens: string[];
  privateTokens: string[];
  phase: 'reserve' | 'attempt' | 'unsafe-probe' | 'revert' | 'safe-probe' | 'committed';
  alarmed: boolean;
  heldInReserve: boolean;
};

export const getMechanismState = (frame: number): MechanismState => {
  if (frame >= MECHANISM_TIMING.commit) {
    return {
      visibleTokens: [...BASE_TOKENS, '18'],
      privateTokens: [],
      phase: 'committed',
      alarmed: false,
      heldInReserve: false,
    };
  }
  if (frame >= MECHANISM_TIMING.secondProbe[0]) {
    return {
      visibleTokens: [...BASE_TOKENS],
      privateTokens: frame >= 345 ? ['18', '.'] : ['18'],
      phase: 'safe-probe',
      alarmed: false,
      heldInReserve: true,
    };
  }
  if (frame >= MECHANISM_TIMING.revert) {
    return {
      visibleTokens: [...BASE_TOKENS],
      privateTokens: [],
      phase: 'revert',
      alarmed: true,
      heldInReserve: true,
    };
  }
  if (frame >= MECHANISM_TIMING.firstProbe[0]) {
    return {
      visibleTokens: [...BASE_TOKENS],
      privateTokens: frame >= 195 ? ['9', 'boxes'] : ['9'],
      phase: 'unsafe-probe',
      alarmed: true,
      heldInReserve: true,
    };
  }
  if (frame >= 72) {
    return {
      visibleTokens: [...BASE_TOKENS],
      privateTokens: [],
      phase: 'attempt',
      alarmed: false,
      heldInReserve: true,
    };
  }
  return {
    visibleTokens: [...BASE_TOKENS],
    privateTokens: [],
    phase: 'reserve',
    alarmed: false,
    heldInReserve: true,
  };
};

const phaseCopy: Record<MechanismState['phase'], {index: string; title: string; detail: string}> = {
  reserve: {index: '01', title: 'Hold', detail: 'Keep the uncompressed cache in reserve.'},
  attempt: {index: '02', title: 'Attempt', detail: 'Apply compression without releasing the reserve.'},
  'unsafe-probe': {index: '03', title: 'Probe privately', detail: 'Two tokens reveal rising causal risk.'},
  revert: {index: '04', title: 'Reject + resume', detail: 'Discard the probe. Continue from the held cache.'},
  'safe-probe': {index: '03', title: 'Probe again', detail: 'This private window remains inside the safety layer.'},
  committed: {index: '05', title: 'Commit', detail: 'Release the reserve only after a safe decision.'},
};

const clamp = {
  easing: EASE,
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const LaneLabel = ({children, accent}: {children: string; accent: string}) => (
  <div
    style={{
      display: 'flex',
      alignItems: 'center',
      gap: 13,
      color: accent,
      fontFamily: FONT.mono,
      fontSize: 18,
      letterSpacing: '0.1em',
    }}
  >
    <span style={{width: 10, height: 10, borderRadius: 10, background: accent}} />
    {children}
  </div>
);

const HeldCacheReserve = ({opacity}: {opacity: number}) => (
  <div
    style={{
      position: 'absolute',
      left: 22,
      top: 0,
      display: 'flex',
      width: 310,
      flexDirection: 'column',
      gap: 8,
      opacity,
    }}
  >
    <span style={{marginBottom: 3, fontFamily: FONT.mono, fontSize: 16, letterSpacing: '0.09em'}}>
      HELD UNCOMPRESSED CACHE
    </span>
    {Array.from({length: 8}, (_, index) => (
      <div
        key={index}
        style={{
          height: 22,
          border: `1px solid ${COLORS.graphite}4D`,
          background: `${COLORS.graphite}08`,
        }}
      />
    ))}
  </div>
);

export const MechanismScene = () => {
  const frame = useCurrentFrame();
  const state = getMechanismState(frame);
  const copy = phaseCopy[state.phase];
  const isUnsafe = state.phase === 'unsafe-probe' || state.phase === 'revert';
  const isSafe = state.phase === 'safe-probe' || state.phase === 'committed';
  const accent = isUnsafe ? COLORS.coral : isSafe ? COLORS.sage : COLORS.blue;
  const risk = isUnsafe
    ? interpolate(frame, [MECHANISM_TIMING.firstProbe[0], MECHANISM_TIMING.revert], [0.42, 0.94], clamp)
    : isSafe
      ? interpolate(frame, [MECHANISM_TIMING.secondProbe[0], MECHANISM_TIMING.commit], [0.29, 0.12], clamp)
      : 0.24;
  const heldOpacity = interpolate(frame, [MECHANISM_TIMING.commit, MECHANISM_TIMING.commit + 28], [1, 0], clamp);
  const revertSweep = interpolate(frame, [MECHANISM_TIMING.revert, MECHANISM_TIMING.revert + 28], [0, 1], clamp);
  const privateOpacity = state.privateTokens.length > 0 ? 1 : 0;

  return (
    <Stage style={{flexDirection: 'column', gap: 34}}>
      <div style={{display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between'}}>
        <div style={{display: 'flex', flexDirection: 'column', gap: 10}}>
          <span style={{color: COLORS.blue, fontFamily: FONT.mono, fontSize: 20, letterSpacing: '0.12em'}}>
            RUNTIME CONTROL LOOP
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
            Two private tokens. One safe decision.
          </h1>
        </div>
        <div
          style={{
            display: 'flex',
            alignItems: 'baseline',
            gap: 13,
            paddingTop: 18,
            color: accent,
            fontFamily: FONT.mono,
          }}
        >
          <span style={{fontSize: 48, lineHeight: 1}}>{copy.index}</span>
          <span style={{fontSize: 20, letterSpacing: '0.1em'}}>OF 05</span>
        </div>
      </div>

      <div style={{display: 'grid', minHeight: 620, flex: 1, gridTemplateColumns: '0.88fr 1.12fr', gap: 92}}>
        <div style={{display: 'flex', flexDirection: 'column', justifyContent: 'space-between'}}>
          <div style={{position: 'relative', height: 390, paddingTop: 18}}>
            <HeldCacheReserve opacity={heldOpacity} />
            <div style={{position: 'absolute', left: 58, top: 62}}>
              <CacheStack
                compression={0.72}
                heldInReserve={false}
                alarmed={state.alarmed}
                metricMode="retained"
              />
            </div>
          </div>

          <div style={{display: 'flex', flexDirection: 'column', gap: 17}}>
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
              <span style={{fontFamily: FONT.mono, fontSize: 18, letterSpacing: '0.1em'}}>CAUSAL RISK</span>
              <span style={{color: accent, fontFamily: FONT.mono, fontSize: 28}}>{risk.toFixed(2)}</span>
            </div>
            <div style={{height: 12, overflow: 'hidden', background: `${COLORS.graphite}14`}}>
              <div style={{width: `${risk * 100}%`, height: '100%', background: accent}} />
            </div>
            <div style={{display: 'flex', flexDirection: 'column', gap: 8}}>
              <strong style={{fontFamily: FONT.sans, fontSize: 36, fontWeight: 600}}>{copy.title}</strong>
              <span style={{maxWidth: 660, color: `${COLORS.graphite}A6`, fontFamily: FONT.sans, fontSize: 26, lineHeight: 1.35}}>
                {copy.detail}
              </span>
            </div>
          </div>
        </div>

        <div style={{display: 'grid', gridTemplateRows: '1fr 1fr', gap: 30}}>
          <div
            style={{
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
              gap: 24,
              overflow: 'hidden',
              padding: '30px 34px',
              border: `2px solid ${accent}`,
              background: `${accent}0B`,
            }}
          >
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
              <LaneLabel accent={accent}>PRIVATE TWO-TOKEN WINDOW</LaneLabel>
              <span style={{fontFamily: FONT.mono, fontSize: 17, letterSpacing: '0.08em'}}>NOT USER-VISIBLE</span>
            </div>
            <div style={{display: 'flex', flex: 1, alignItems: 'center'}}>
              {state.privateTokens.length > 0 ? (
                <div
                  style={{
                    opacity: privateOpacity,
                    translate: `${isUnsafe ? interpolate(revertSweep, [0, 1], [0, -180]) : 0}px 0`,
                  }}
                >
                  <TokenStream tokens={state.privateTokens} state={isUnsafe ? 'unstable' : 'probe'} />
                </div>
              ) : (
                <span style={{color: `${COLORS.graphite}66`, fontFamily: FONT.mono, fontSize: 25}}>
                  {state.phase === 'revert' ? 'PROBE DISCARDED' : state.phase === 'committed' ? 'WINDOW CLEARED' : 'AWAITING PROBE'}
                </span>
              )}
            </div>
            {isUnsafe ? (
              <div
                style={{
                  position: 'absolute',
                  right: 28,
                  bottom: 25,
                  color: COLORS.coral,
                  fontFamily: FONT.mono,
                  fontSize: 18,
                  letterSpacing: '0.1em',
                }}
              >
                REJECT PATH
              </div>
            ) : null}
          </div>

          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 24,
              padding: '30px 34px',
              border: `1px solid ${COLORS.graphite}24`,
              background: COLORS.paper,
            }}
          >
            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
              <LaneLabel accent={COLORS.graphite}>USER-VISIBLE OUTPUT</LaneLabel>
              <span style={{color: COLORS.sage, fontFamily: FONT.mono, fontSize: 17, letterSpacing: '0.08em'}}>
                COMMITTED TOKENS ONLY
              </span>
            </div>
            <div style={{display: 'flex', flex: 1, alignItems: 'center'}}>
              <TokenStream tokens={state.visibleTokens} state={state.phase === 'committed' ? 'committed' : 'stable'} />
            </div>
          </div>
        </div>
      </div>
    </Stage>
  );
};
