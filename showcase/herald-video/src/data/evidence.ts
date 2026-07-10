import {z} from 'zod';
import raw from './evidence.generated.json';

const MetricRow = z.object({
  episodes: z.literal(184),
  compressed_generation_fraction: z.number().min(0).max(1),
  quality_cost: z.number(),
  revert_wall_overhead: z.number().nonnegative(),
  token_overhead: z.number().nonnegative(),
});

export const EvidenceSchema = z.object({
  campaign: z.object({
    episode_count: z.literal(552),
    prompt_count: z.literal(46),
    ratio_count: z.literal(4),
    compressor_count: z.literal(3),
    compressors: z.object({
      expected_attention: MetricRow,
      knorm: MetricRow,
      streaming_llm: MetricRow,
    }),
  }),
  gsm8k_example: z.object({
    prompt_id: z.literal('gsm8k-0'),
    ratio: z.literal(0.75),
    switch_position: z.literal(128),
    reference_answer: z.literal('18'),
    compressed_answer: z.literal('3'),
    reference_quality: z.literal(1),
    compressed_quality: z.literal(0),
    reference_excerpt: z.string(),
    compressed_excerpt: z.string(),
  }),
});

export type Evidence = z.infer<typeof EvidenceSchema>;
export const evidence: Evidence = EvidenceSchema.parse(raw);
