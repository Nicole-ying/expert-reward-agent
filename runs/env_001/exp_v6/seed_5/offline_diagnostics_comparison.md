# Seed 5 Offline Diagnostics Comparison

> Shadow-mode analysis. All iterations replay the saved deterministic policy on LunarLander-v2 with the same 20 evaluation seeds (10005-10024). No policy was retrained. Component statistics describe evaluation trajectories, not historical training trajectories.

## External Behavior

| iteration | original return mean | return std | episode length mean | terminated | truncated | final distance mean | final speed mean | both-contact rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -111.22 | 8.64 | 68.25 | 20/20 | 0/20 | 0.285 | 0.566 | 1.9% |
| 2 | -94.01 | 21.61 | 68.90 | 20/20 | 0/20 | 0.287 | 0.510 | 1.9% |
| 3 | 238.08 | 42.16 | 452.10 | 18/20 | 2/20 | 0.062 | 0.0001 | 18.5% |
| 4 | 147.51 | 21.54 | 1000.00 | 0/20 | 20/20 | 0.040 | 0.0040 | 73.1% |

`terminated` is an observed Gymnasium flag and is not automatically labeled success or crash. `truncated` corresponds to the 1000-step time limit in these replays.

## Per-Episode Reward Contribution

| iteration | progress sum mean | landing proxy sum mean | stability sum mean | generated return mean |
|---:|---:|---:|---:|---:|
| 1 | 1.125 | 0.350 | -0.764 | 0.711 |
| 2 | 1.123 | 0.408 | -0.763 | 0.768 |
| 3 | 1.348 | 119.798 | -0.313 | 120.833 |
| 4 | 6.848 | 716.749 | -0.363 | 723.234 |

## Intervention Checks

### Iter1 to Iter2: binary proxy to continuous proxy

Observed facts:

- Landing proxy activation increased from 0.51% to 1.23% of evaluation steps.
- Mean episode length remained effectively unchanged: 68.25 to 68.90.
- Final distance remained effectively unchanged: 0.285 to 0.287.
- Final speed improved modestly: 0.566 to 0.510.
- External return improved from -111.22 to -94.01 but remained far below the target.

Conclusion: continuous proxy made the terminal-quality signal somewhat more available, but it did not repair the dominant failure behavior. This modification alone was insufficient.

### Iter2 to Iter3: global stability penalty to distance-gated stability penalty

Observed facts:

- Mean per-episode stability cost decreased in magnitude from -0.763 to -0.313.
- Mean episode length increased from 68.90 to 452.10.
- Final distance decreased from 0.287 to 0.062.
- Final speed decreased from 0.510 to approximately zero.
- Both-contact rate increased from 1.9% to 18.5%.
- External return increased from -94.01 to 238.08.
- The legacy flat output is arithmetically consistent with `progress + landing_proxy + stability`; `dist_gate` is a non-additive modulator, with mean residual below 1e-17.

Supported interpretation: applying the stability constraint conditionally near the target removed a strong global conflict and enabled a qualitatively different policy. Because each reward was trained once, this is strong mechanism evidence but not yet a causal estimate across training seeds.

### Iter3 to Iter4: progress coefficient increased from 1 to 5

Observed facts:

- All 20 episodes reached the 1000-step time limit instead of environment termination.
- Final state remained near the target, slow and nearly upright.
- Both-contact rate increased from 18.5% to 73.1%.
- Landing proxy accumulated 716.75 per episode on average.
- External return fell from 238.08 to 147.51 despite generated return increasing from 120.83 to 723.23.

Supported interpretation: the revised reward strongly favors a long-lived, proxy-active behavior that is misaligned with prompt environment completion. The replay cannot determine the exact physical termination failure (for example persistent micro-motion) because that reason is not exposed by the environment interface. It therefore does not label this behavior as reward hacking or a crash.

## What The New Statistics Add

The old global per-step means show component scale but hide episode outcomes. The added per-episode sums and behavior facts distinguish three cases that previously looked similar:

1. Iter1/2: short terminated trajectories with high final speed and almost no contact.
2. Iter3: near-target, near-zero-speed trajectories that usually terminate before the time limit.
3. Iter4: near-target, contact-heavy trajectories that always remain alive until the time limit while accumulating proxy reward.

This supports adding termination/truncation, per-episode component sums, final-state metrics and contact/action statistics to future feedback. It does not yet establish that giving all of these fields to the reflection LLM improves search; that requires the later component-only versus enhanced-feedback ablation.
