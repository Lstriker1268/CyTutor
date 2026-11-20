# Llama 3 8B vs. 70B Baseline Evaluation

## Overview

This preliminary experiment compares Llama 3 8B and Llama 3 70B on five TCP three-way handshake questions. Both configurations use greedy decoding, a maximum of 60 new tokens, and a repetition penalty of 1.1.

## Results

| Metric | Llama 3 8B (v3) | Llama 3 70B (v4) |
|---|---:|---:|
| Questions evaluated | 5 | 5 |
| Mean sentence BLEU-4 | 0.0697 | 0.1443 |
| Mean generation time | 42.22 seconds | 4.20 seconds |
| Total evaluation-loop runtime | 211.133 seconds | 21.012 seconds |
| Peak GPU memory allocated | 12.945 GiB | 36.838 GiB |
| GPU | NVIDIA Tesla V100 16GB | NVIDIA A100 80GB |
| Precision | FP16 | 4-bit NF4, BF16 compute |

## Initial Findings

The 70B configuration achieved a mean sentence BLEU-4 score of 0.1443, compared with 0.0697 for the 8B configuration. This is an absolute increase of 0.0746, or approximately 107% relative to the 8B score.

The observed mean generation time was approximately 4.20 seconds for the 70B configuration and 42.22 seconds for the 8B configuration. These measurements compare complete configurations, not model size alone, because the experiments used different GPUs and precision settings.

## Limitations

- The evaluation contains only five questions from one networking topic.
- Sentence BLEU-4 measures word and phrase overlap, not factual correctness or educational usefulness.
- The small sample size means the results are preliminary and may not generalize to other cybersecurity topics.
- GPU hardware and precision differ between runs, so the timing comparison does not isolate model-size effects.
- The generated answers should also be reviewed for factual accuracy, completeness, and clarity.

## Next Steps

1. Manually score each answer for correctness, completeness, and clarity.
2. Expand the evaluation set to include DNS, HTTP, Linux, and additional cybersecurity topics.
3. Re-run comparisons with a consistent evaluation protocol and clearly documented hardware settings.
