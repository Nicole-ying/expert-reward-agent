# Iteration Context

## Recommended Action
**tune** — 当前骨架仅运行1轮，得分-102.55远低于目标200。stability_penalty 出现极大负值（-334.8），主导总奖励，导致agent不敢动。progress_delta_reward 均值接近0，驱动力不足。soft_landing_bonus 从未触发，条件过严。best_reward.py与当前代码完全相同，无需revert。建议：降低stability_penalty权重（如从0.5/0.3降至0.1/0.05），增大progress_delta_reward权重（如从2.0增至5.0），放宽soft_landing_bonus条件（如降低速度下限至0.2）。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -102.55 | -102.55 | 0.00 | 33.00 | energy_penalty=-0.017 progress_delta_reward=0.003 soft_landing_bonus=0.000 stability_penalty=-0.137 | new_best |

## Expert Cards
## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## generated_reward_negative_average
- signal: total generated reward mean is negative during most training steps
- risk: agent receives mostly punitive feedback and may not discover useful behavior
- fix: rebalance progress and penalty terms; avoid large always-on penalties in early versions

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.016592 | 0.016592 | 1.000000 | -0.040000 | -0.000005 |
| progress_delta_reward | 0.002806 | 0.037654 | 1.000000 | -0.806997 | 0.792863 |
| soft_landing_bonus | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.500000 |
| stability_penalty | -0.136703 | 0.136703 | 1.000000 | -334.813210 | -0.000015 |
| total_reward | -0.147752 | 0.153141 | 1.000000 | -20.000000 | 0.414194 |
| generated_reward | -0.147752 | 0.153141 | 1.000000 | -20.000000 | 0.414194 |
| original_env_reward | -1.569467 | 1.592469 | 1.000000 | -100.000000 | 0.561919 |
