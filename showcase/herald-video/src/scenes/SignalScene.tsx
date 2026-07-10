import {interpolate, useCurrentFrame} from 'remotion';
import {SignalTrace} from '../components/SignalTrace';
import {Stage} from '../components/Stage';
import {TokenStream} from '../components/TokenStream';
import {COLORS, EASE, FONT} from '../theme';

const SIGNALS = [
  {label: 'ENTROPY', values: [0.24, 0.3, 0.27, 0.38, 0.44, 0.51, 0.49, 0.66], accent: 'blue' as const},
  {label: 'CONFIDENCE MARGIN', values: [0.72, 0.68, 0.63, 0.57, 0.49, 0.42, 0.31, 0.25], accent: 'coral' as const},
  {label: 'KL CHANGE', values: [0.12, 0.14, 0.13, 0.19, 0.22, 0.41, 0.47, 0.59], accent: 'coral' as const},
  {label: 'ROLLING DYNAMICS', values: [0.31, 0.33, 0.29, 0.38, 0.35, 0.46, 0.52, 0.56], accent: 'blue' as const},
] as const;

const clamp = {
  easing: EASE,
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

export const SignalScene = () => {
  const frame = useCurrentFrame();
  const planeExit = interpolate(frame, [72, 126], [0, 1], clamp);
  const traceProgress = interpolate(frame, [108, 306], [0, 1], clamp);
  const riskBuild = interpolate(frame, [312, 376], [0, 1], clamp);
  const finalFocus = interpolate(frame, [330, 390], [0, 1], clamp);

  return (
    <Stage style={{flexDirection: 'column', gap: 38}}>
      <div style={{display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between'}}>
        <div style={{display: 'flex', maxWidth: 830, flexDirection: 'column', gap: 12}}>
          <span
            style={{
              color: COLORS.blue,
              fontFamily: FONT.mono,
              fontSize: 20,
              letterSpacing: '0.12em',
            }}
          >
            CAUSAL LOGIT SIGNALS
          </span>
          <h1
            style={{
              margin: 0,
              fontFamily: FONT.display,
              fontSize: 82,
              fontWeight: 400,
              letterSpacing: '-0.035em',
              lineHeight: 0.98,
            }}
          >
            The warning is already inside the model.
          </h1>
        </div>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 13,
            paddingTop: 18,
            color: COLORS.graphite,
            fontFamily: FONT.mono,
            fontSize: 20,
            letterSpacing: '0.08em',
          }}
        >
          <span style={{width: 10, height: 10, borderRadius: 10, background: COLORS.sage}} />
          DURING DECODING
        </div>
      </div>

      <div style={{display: 'grid', minHeight: 640, flex: 1, gridTemplateColumns: '0.76fr 1.24fr', gap: 94}}>
        <div style={{position: 'relative', display: 'flex', flexDirection: 'column', justifyContent: 'center'}}>
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 30,
              opacity: 1 - planeExit,
              translate: `${interpolate(planeExit, [0, 1], [0, -50])}px 0`,
            }}
          >
            <div style={{fontFamily: FONT.sans, fontSize: 28, fontWeight: 500}}>GENERATED TOKEN PLANE</div>
            <TokenStream tokens={['Money', 'made', '=', '9', '×', '2', '=', '$18']} state="stable" />
            <div style={{height: 1, background: `${COLORS.graphite}28`}} />
            <p
              style={{
                margin: 0,
                maxWidth: 560,
                color: `${COLORS.graphite}A6`,
                fontFamily: FONT.sans,
                fontSize: 28,
                lineHeight: 1.36,
              }}
            >
              Each next-token decision already produces the statistics HERALD observes.
            </p>
          </div>

          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              gap: 22,
              opacity: finalFocus,
              translate: `${interpolate(finalFocus, [0, 1], [26, 0])}px 0`,
            }}
          >
            <div style={{fontFamily: FONT.mono, fontSize: 20, letterSpacing: '0.12em'}}>ONE CAUSAL RISK SCORE</div>
            <div style={{display: 'flex', alignItems: 'center', gap: 26}}>
              <span
                style={{
                  fontFamily: FONT.display,
                  fontSize: 112,
                  letterSpacing: '-0.05em',
                  lineHeight: 0.9,
                }}
              >
                HIGH
              </span>
              <div style={{display: 'flex', height: 190, width: 28, alignItems: 'flex-end', background: `${COLORS.graphite}12`}}>
                <div style={{width: '100%', height: `${riskBuild * 100}%`, background: COLORS.coral}} />
              </div>
            </div>
            <span style={{color: COLORS.coral, fontFamily: FONT.sans, fontSize: 31, fontWeight: 600}}>
              RISK RISING
            </span>
          </div>
        </div>

        <div style={{display: 'grid', gridTemplateRows: 'repeat(4, 1fr)', gap: 6}}>
          {SIGNALS.map((signal, index) => {
            const rowProgress = interpolate(traceProgress, [index * 0.18, 0.46 + index * 0.18], [0, 1], {
              extrapolateLeft: 'clamp',
              extrapolateRight: 'clamp',
            });
            return (
              <div
                key={signal.label}
                style={{
                  height: 144,
                  overflow: 'hidden',
                  opacity: interpolate(rowProgress, [0, 0.2], [0, 1], {
                    extrapolateLeft: 'clamp',
                    extrapolateRight: 'clamp',
                  }),
                  translate: `${interpolate(rowProgress, [0, 1], [36, 0])}px 0`,
                }}
              >
                <div style={{scale: 0.78, transformOrigin: 'top left'}}>
                  <SignalTrace {...signal} values={[...signal.values]} progress={rowProgress} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div
        style={{
          position: 'absolute',
          right: 120,
          bottom: 74,
          display: 'flex',
          alignItems: 'center',
          gap: 18,
          opacity: interpolate(frame, [326, 354], [0, 1], clamp),
          color: COLORS.blue,
          fontFamily: FONT.mono,
          fontSize: 25,
          letterSpacing: '0.09em',
        }}
      >
        <span style={{width: 52, height: 2, background: COLORS.blue}} />
        NO ADDITIONAL FORWARD PASS
      </div>
    </Stage>
  );
};
