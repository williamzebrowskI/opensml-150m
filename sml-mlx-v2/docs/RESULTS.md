# Results and Evaluation

**Interim snapshot | 2026-09-18 | Through step 7,982**

[Overview](../README.md) · [Report](TECHNICAL_REPORT.md) · [Evidence](EVIDENCE.md)

> This page is dated, not live. Later checkpoints can supersede these values.
> The 15B-token target has not been reached in this snapshot.

## Evaluation Protocol

Mac-1 evaluates 16 fixed batches per source, each with two 2,048-token sequences:
64 batches and 262,144 target positions overall. Per-source mean losses are
combined with 55/25/10/10 weights, not equal source weights. Perplexity is the
exponential of aggregate token loss; it is not answer accuracy.

Restart reconstructs held-out batches and checks their fingerprint. Pilot and
continuation share this validation identity:

```text
99e7d4417061db5a70138701694ad5577d0f9ff0d0fc729ddc6beff06b3a7081
```

## Recorded Milestones

These are selected milestones, **not the full evaluation curve**. Include all
intermediate evaluations before publishing a full learning-curve figure.

| Event | Step | Total tokens | Validation loss | Perplexity |
| --- | ---: | ---: | ---: | ---: |
| Pilot endpoint; continuation parent | 4,696 | 500,105,216 | 3.3536 | 28.61 |
| Continuation near end of LR rewarm | 5,165 | 550,051,840 | 3.4327 | 30.96 |
| Continuation best at snapshot | 7,982 | 850,051,072 | 3.3160 | 27.55 |

Loss rose during rewarm, then fell below the pilot endpoint. This is one run's
trajectory, not a controlled demonstration of the LR policy's benefit. Total
continuation counts already include the pilot; do not add its tokens twice.

### Per-source Loss at Step 7,982

| Web | DCLM | Wiki | Cosmopedia |
| ---: | ---: | ---: | ---: |
| 3.4202 | 3.6129 | 2.8130 | 2.5034 |

Source difficulty differs; a lower value does not by itself establish stronger
general capability in that domain.

### Throughput Observation

Step 7,980 logged 13,794 tokens/sec for the logging interval and 13,578 run
tokens/sec. These are production counters, not a repeated benchmark, utilization
measurement, or certified total training duration. Interval timing can include
evaluation/checkpoint pauses; resumed jobs have their own timing windows.
V1's roughly 43K throughput is not a result for this deeper, longer-context V2.

## Generation Review

The trainer uses five fixed prose prompts, greedy decoding (`temperature=0`),
and up to 64 new tokens, with earlier termination on EOS. Prompts are in the
recipe; decoding is in the trainer. Playground settings are separate observations.

Pilot and early-continuation samples exhibit repetition, weak factual grounding,
and incoherence. Improving loss does not establish that these problems are solved.
No generation success percentage is asserted here.

Future reviews should retain every query and full completion, checkpoint identity,
decoding settings, seed, stopping reason, and rubric. Include failures and separate
inspected development prompts from an untouched final suite.

## Not Yet Reported

- Completed 15B outcome and final checkpoint selection.
- Full learning curves exported across jobs with restart/schedule boundaries.
- Independent downstream benchmarks and contamination checks.
- A fixed full-completion review with quality and repetition metrics.
- V2 SFT or preference-training outcomes.
- Repeated matched performance experiments and independently measured peak memory.

See [Evidence](EVIDENCE.md) for artifact locations and update rules.
