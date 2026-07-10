import {expect, it} from 'vitest';
import {EvidenceSchema, evidence} from '../data/evidence';

it('loads only artifact-backed campaign evidence', () => {
  expect(evidence.campaign.episode_count).toBe(552);
  expect(evidence.gsm8k_example).toMatchObject({
    prompt_id: 'gsm8k-0',
    reference_answer: '18',
    compressed_answer: '3',
  });
});

it('rejects campaign rows without exactly 184 episodes', () => {
  expect(EvidenceSchema).toBeDefined();
  const invalid = structuredClone(evidence) as unknown as {
    campaign: {
      compressors: {expected_attention: {episodes: number}};
    };
  };
  invalid.campaign.compressors.expected_attention.episodes = 183;

  expect(() => EvidenceSchema.parse(invalid)).toThrow();
});
