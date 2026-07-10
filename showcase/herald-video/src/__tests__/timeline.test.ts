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
