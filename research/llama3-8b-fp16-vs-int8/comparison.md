# Llama 3 8B: FP16 vs. INT8 Evaluation

## Overview

This experiment compares Meta Llama 3 8B in unquantized FP16 precision against the same model using 8-bit INT8 quantization with bitsandbytes.

Both experiments used the same five-question networking evaluation dataset and generation settings.

## Experimental Setup

| Setting | Value |
|---|---|
| Model | meta-llama/Meta-Llama-3-8B |
| Dataset | networking_eval.jsonl |
| Number of questions | 5 |
| Maximum new tokens | 60 |
| Sampling | Disabled |
| Repetition penalty | 1.1 |
| GPU | NVIDIA Tesla V100-SXM2-16GB |
| FP16 precision | FP16, unquantized |
| INT8 precision | 8-bit INT8, bitsandbytes |

## Results

| Metric | FP16 | INT8 |
|---|---:|---:|
| Total runtime | 211.133 seconds | 65.107 seconds |
| Average generation time | 42.225 seconds/question | 13.020 seconds/question |
| Peak GPU memory allocated | 12.945 GiB | 8.486 GiB |
| Questions evaluated | 5 | 5 |

## Performance Differences

- **Total runtime:** INT8 completed the evaluation in 69.2% less time.
- **GPU memory:** INT8 used 4.459 GiB less peak allocated GPU memory, a reduction of 34.4%.
- **Average generation speed:** INT8 was approximately 3.24 times faster based on the reported average generation times.

## Per-Question Generation Times

| Question | FP16 | INT8 |
|---|---:|---:|
| 1 | 62.843 s | 13.484 s |
| 2 | 2.904 s | 13.117 s |
| 3 | 33.676 s | 12.937 s |
| 4 | 55.933 s | 12.610 s |
| 5 | 55.768 s | 12.953 s |

## Interpretation

In this five-question experiment, INT8 had a lower total runtime and lower peak PyTorch GPU memory allocation than FP16.

However, FP16 generation times varied considerably between questions, from 2.904 to 62.843 seconds. INT8 generation times were more consistent, ranging from 12.610 to 13.484 seconds. This variability means the runtime difference should be interpreted cautiously.

Repeated runs, warm-up tests, and consistent timing procedures would help establish a more reliable performance comparison.

The memory measurements represent peak GPU memory allocated by PyTorch, not necessarily total GPU memory used by the process.

## Answer Quality and Limitations

This experiment evaluates runtime and memory usage, not answer accuracy.

It does not establish whether FP16 or INT8 produces more accurate, complete, or useful cybersecurity answers. A follow-up experiment should compare both sets of answers using the same evaluation metrics and, ideally, manual review.

If BLEU-4 is used, it should be treated as a text-overlap metric rather than a direct measure of factual correctness.

The evaluation contains only five questions, so these results are preliminary and should not be generalized to all cybersecurity tasks.

## Conclusion

On the five-question networking evaluation using an NVIDIA Tesla V100 16GB GPU, Llama 3 8B INT8 completed the evaluation in 65.107 seconds and used 8.486 GiB of peak allocated GPU memory. The FP16 run took 211.133 seconds and used 12.945 GiB.

INT8 therefore showed better measured runtime and memory efficiency in this experiment. Further repeated testing and answer-quality evaluation are needed before drawing broader conclusions.

## Reproducibility

The experiment files are located in `~/cytutor-baseline/`.

FP16 results:
- `baseline_results_v3.jsonl`
- `baseline_results_v3_summary.json`

INT8 results:
- `baseline_results_v5_int8.jsonl`
- `baseline_results_v5_int8_summary.json`
