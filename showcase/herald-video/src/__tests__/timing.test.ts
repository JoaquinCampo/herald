import {expect, it} from 'vitest';
import {MECHANISM_TIMING, getMechanismState} from '../scenes/MechanismScene';

it('keeps both probe tokens private before revert or commit', () => {
  expect(MECHANISM_TIMING.firstProbe).toEqual([180, 210]);
  expect(MECHANISM_TIMING.revert).toBe(224);
  expect(MECHANISM_TIMING.secondProbe).toEqual([330, 360]);
  expect(MECHANISM_TIMING.commit).toBe(374);
});

it('never exposes rejected probe tokens in the user-visible lane', () => {
  for (let frame = 0; frame < MECHANISM_TIMING.commit; frame += 1) {
    expect(getMechanismState(frame).visibleTokens).toEqual(['The', 'answer', 'is']);
  }

  expect(getMechanismState(MECHANISM_TIMING.commit).visibleTokens).toEqual([
    'The',
    'answer',
    'is',
    '18',
    '.',
  ]);
});

it('resumes uncompressed after rejection before attempting compression again', () => {
  expect(getMechanismState(0)).toMatchObject({
    phase: 'reserve',
    compressionActive: false,
    reserveMode: 'active',
  });
  expect(getMechanismState(72)).toMatchObject({
    phase: 'attempt',
    compressionActive: true,
    reserveMode: 'held',
  });
  expect(getMechanismState(MECHANISM_TIMING.revert + 26)).toMatchObject({
    phase: 'resume',
    compressionActive: false,
    reserveMode: 'active',
    privateTokens: [],
  });
  expect(getMechanismState(300)).toMatchObject({
    phase: 'second-attempt',
    compressionActive: true,
    reserveMode: 'held',
  });
});

it('visibly reverses rejected tokens only inside the private lane', () => {
  const reversal = getMechanismState(MECHANISM_TIMING.revert + 12);
  expect(reversal).toMatchObject({
    phase: 'revert',
    privateTokens: ['9', 'boxes'],
    visibleTokens: ['The', 'answer', 'is'],
    reverseProgress: 12 / 26,
  });
});

it('commits both accepted private tokens and releases the reserve', () => {
  expect(getMechanismState(355)).toMatchObject({
    phase: 'safe-probe',
    privateTokens: ['18', '.'],
    visibleTokens: ['The', 'answer', 'is'],
  });
  expect(getMechanismState(MECHANISM_TIMING.commit)).toMatchObject({
    phase: 'committed',
    privateTokens: [],
    visibleTokens: ['The', 'answer', 'is', '18', '.'],
    compressionActive: true,
    reserveMode: 'released',
  });
});
