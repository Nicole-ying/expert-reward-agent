# Iteration Context

## Recommended Action
**tune** — 当前骨架首次迭代即达到302.9分，远超目标200分，且best_reward与previous_reward代码完全相同，无需revert。组件信号显示forward_reward强正，angle_penalty中等负，其他弱。得分高但可能存在goal_near_oscillation风险（forward_reward指数形式在目标附近震荡）和high_reward_without_success（无成功检测）。建议小幅调整forward_reward温度或添加软着陆代理以稳定行为。

## Agent Memory
| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + angular_penalty + energy_penalty + forward_reward | 302.90 | 302.90 | 0.00 | 994.80 | angle_penalty=-0.053 angular_penalty=-0.000 energy_penalty=-0.020 forward_reward=1.043 | target_solved_new_best |

## Expert Cards
## goal_near_oscillation
- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## high_reward_without_success
- signal: generated_reward improves but external score stays poor
- risk: policy optimizes the custom reward but not the real task
- fix: reduce exploitable terms; add constraints tied to actual task progress and stable outcome

## Stable Lessons (from previous iterations)
- Target: mean external score >= 200.
- terminal_success_reward and terminal_failure_penalty blocked (no explicit flag).
- Use external evaluation as fitness signal, not generated_reward alone.
- Contact only inside guarded landing proxy (near target + low speed + stable angle).
- Prefer continuous shaping over hard sparse bonuses.

## Component Evidence
| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| angle_penalty | -0.053168 | 0.053168 | 0.999997 | -5.011347 | -0.000000 |
| angular_penalty | -0.000464 | 0.000464 | 0.999955 | -0.014550 | -0.000000 |
| energy_penalty | -0.020373 | 0.020373 | 1.000000 | -0.040000 | -0.000015 |
| forward_reward | 1.043373 | 1.048752 | 1.000000 | -0.525032 | 4.450016 |
| total_reward | 0.969368 | 0.985428 | 1.000000 | -5.170992 | 4.271336 |
| generated_reward | 0.969368 | 0.985428 | 1.000000 | -5.170992 | 4.271336 |
| original_env_reward | -0.063751 | 0.422043 | 1.000000 | -100.000000 | 0.970187 |
