# Iteration Context

## Recommended Action
**tune** — 当前得分11.95远低于目标200，但相比上一轮-102.55有显著提升。best_reward与previous_reward代码完全相同，无需revert。骨架仅迭代2轮，尚未停滞。主要问题：progress_delta_reward系数5.0可能不足，建议提升至10-20；stability_penalty过弱，建议增大角度和角速度惩罚系数；soft_landing_bonus稀疏且条件宽松，可能引发contact_reward_hacking，建议收紧条件或降低权重。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -102.55 | -102.55 | 0.00 | 33.00 | energy_penalty=-0.017 progress_delta_reward=0.003 soft_landing_bonus=0.000 stability_penalty=-0.137 | new_best |
| 2 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | 11.95 | 11.95 | 0.00 | 863.30 | energy_penalty=-0.015 progress_delta_reward=0.012 soft_landing_bonus=0.073 stability_penalty=-0.004 | new_best |

## Expert Cards
## high_reward_without_success
- signal: generated_reward improves but external score stays poor
- risk: policy optimizes the custom reward but not the real task
- fix: reduce exploitable terms; add constraints tied to actual task progress and stable outcome

## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

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

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| energy_penalty | -0.015452 | 0.015452 | 1.000000 | -0.040000 | -0.000008 |
| progress_delta_reward | 0.011617 | 0.080649 | 1.000000 | -2.235938 | 1.531983 |
| soft_landing_bonus | 0.072954 | 0.072954 | 0.145908 | 0.000000 | 0.500000 |
| stability_penalty | -0.004027 | 0.004027 | 0.261542 | -0.296466 | -0.000000 |
| total_reward | 0.065092 | 0.148332 | 1.000000 | -2.530425 | 1.980922 |
| generated_reward | 0.065092 | 0.148332 | 1.000000 | -2.530425 | 1.980922 |
| original_env_reward | -0.635072 | 0.837494 | 1.000000 | -100.000000 | 0.927277 |
