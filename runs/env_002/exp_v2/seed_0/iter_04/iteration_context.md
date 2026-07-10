# Iteration Context

## Recommended Action
**revert** — 当前得分-76.01远低于best得分11.95，差距87.96。对比代码，current将progress_delta_reward系数从5.0提升至15.0，但信号仍弱且出现大幅负值；stability_penalty系数和阈值收紧但惩罚过弱；soft_landing_bonus权重降低且条件收紧导致几乎不触发。这些修改均导致性能下降。建议恢复到best_reward的系数（progress_delta_reward=5.0, stability_penalty系数0.1/0.05, soft_landing_bonus=0.5且条件放宽），仅在此基础上小幅调整。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -102.55 | -102.55 | 0.00 | 33.00 | energy_penalty=-0.017 progress_delta_reward=0.003 soft_landing_bonus=0.000 stability_penalty=-0.137 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | 11.95 | 11.95 | 0.00 | 863.30 | energy_penalty=-0.015 progress_delta_reward=0.012 soft_landing_bonus=0.073 stability_penalty=-0.004 | new_best |
| 3 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -76.01 | 11.95 | -87.96 | 104.40 | energy_penalty=-0.016 progress_delta_reward=0.051 soft_landing_bonus=0.008 stability_penalty=-0.006 | no_meaningful_improvement |

## Expert Cards
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## high_reward_without_success
- signal: generated_reward improves but external score stays poor
- risk: policy optimizes the custom reward but not the real task
- fix: reduce exploitable terms; add constraints tied to actual task progress and stable outcome

## contact_reward_hacking
- signal: contact-related reward triggers without good external score
- risk: agent exploits contact flags without safe task completion
- fix: require near target, low speed, stable angle, and both supports before using contact

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.
- progress_delta_reward coefficient should be >= 5 to drive learning
- stability_penalty should be stronger to prevent instability

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.015598 | 0.015598 | 1.000000 | -0.040000 | -0.000004 |
| progress_delta_reward | 0.051131 | 0.273950 | 0.999999 | -10.290904 | 4.781192 |
| soft_landing_bonus | 0.008465 | 0.008465 | 0.042326 | 0.000000 | 0.200000 |
| stability_penalty | -0.005643 | 0.005643 | 0.142118 | -0.889759 | -0.000000 |
| total_reward | 0.038355 | 0.279512 | 1.000000 | -10.375505 | 4.740382 |
| generated_reward | 0.038355 | 0.279512 | 1.000000 | -10.375505 | 4.740382 |
| original_env_reward | -0.991363 | 1.235989 | 1.000000 | -100.000000 | 0.795828 |
