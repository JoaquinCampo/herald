import {COLORS, FONT} from '../theme';

export const EvidenceLabel = () => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 12,
      color: COLORS.graphite,
      fontFamily: FONT.mono,
      fontSize: 20,
      letterSpacing: '0.1em',
      lineHeight: 1,
    }}
  >
    <span style={{width: 9, height: 9, borderRadius: '50%', backgroundColor: COLORS.coral}} />
    LIVE LLAMA · IFEVAL · ORION
  </div>
);
