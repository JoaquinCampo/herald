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
  ]);
});
