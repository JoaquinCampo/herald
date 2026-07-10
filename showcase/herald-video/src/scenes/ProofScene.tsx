import {interpolate, useCurrentFrame} from 'remotion';
import {EvidenceLabel} from '../components/EvidenceLabel';
import {MetricLane} from '../components/MetricLane';
import {Stage} from '../components/Stage';
import {evidence} from '../data/evidence';
import {COLORS, EASE, FONT} from '../theme';

export const proofCopy =
  'Live-internal compressed-generation fraction · live-internal quality cost · reverted-attempt wall overhead';

export const formatPercent = (value: number, digits: number) => `${(value * 100).toFixed(digits)}%`;

export const formatQualityPoints = (value: number) => `${(value * 100).toFixed(2)} pp`;

const clamp = {
  easing: EASE,
  extrapolateLeft: 'clamp' as const,
  extrapolateRight: 'clamp' as const,
};

const CAMPAIGN_LANES = [
  ['EXPECTEDATTENTION', evidence.campaign.compressors.expected_attention],
  ['KNORM', evidence.campaign.compressors.knorm],
  ['STREAMINGLLM', evidence.campaign.compressors.streaming_llm],
] as const;

export const ProofScene = () => {
  const frame = useCurrentFrame();
  const intro = interpolate(frame, [0, 24], [0, 1], clamp);
  const detail = interpolate(frame, [410, 438], [0, 1], clamp);

  return (
    <Stage style={{flexDirection: 'column', gap: 34}}>
      <div style={{display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between'}}>
        <div style={{display: 'flex', flexDirection: 'column', gap: 13}}>
          <span style={{color: COLORS.blue, fontFamily: FONT.mono, fontSize: 20, letterSpacing: '0.12em'}}>
            LIVE CONTROLLER CAMPAIGN
          </span>
          <h1
            style={{
              margin: 0,
              fontFamily: FONT.display,
              fontSize: 84,
              fontWeight: 400,
              letterSpacing: '-0.04em',
              lineHeight: 0.98,
            }}
          >
            HERALD adapts its caution to the compressor.
          </h1>
        </div>
        <EvidenceLabel />
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 20,
          opacity: intro,
          translate: `0 ${interpolate(intro, [0, 1], [20, 0])}px`,
        }}
      >
        <span style={{fontFamily: FONT.display, fontSize: 124, letterSpacing: '-0.06em', lineHeight: 0.82}}>
          {evidence.campaign.episode_count}
        </span>
        <span style={{fontFamily: FONT.mono, fontSize: 23, letterSpacing: '0.1em'}}>LIVE HELD-OUT EPISODES</span>
      </div>

      <div style={{display: 'flex', flexDirection: 'column', flex: 1, justifyContent: 'flex-end'}}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1.25fr 0.7fr 2.2fr 0.75fr 0.75fr',
            gap: 28,
            paddingBottom: 14,
            color: `${COLORS.graphite}A6`,
            fontFamily: FONT.mono,
            fontSize: 15,
            letterSpacing: '0.07em',
          }}
        >
          <span>COMPRESSOR</span>
          <span>FRACTION</span>
          <span>COMPRESSED GENERATION</span>
          <span>QUALITY COST</span>
          <span>ROLLBACK</span>
        </div>
        {CAMPAIGN_LANES.map(([name, values], index) => (
          <MetricLane
            key={name}
            name={name}
            fraction={values.compressed_generation_fraction}
            qualityCost={values.quality_cost}
            overhead={values.revert_wall_overhead}
            delay={70 + index * 56}
          />
        ))}
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 17,
          opacity: detail,
          color: COLORS.graphite,
          fontFamily: FONT.mono,
          fontSize: 17,
          letterSpacing: '0.055em',
        }}
      >
        <span style={{width: 48, height: 2, background: COLORS.coral}} />
        {proofCopy.toUpperCase()}
      </div>
    </Stage>
  );
};
