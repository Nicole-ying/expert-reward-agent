# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: 0.620

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| angle_penalty + landing_quality_reward + potential_diff | 2 | 0.620 | -16.610 | unsolved |
| angle_penalty + potential_diff | 1 | -24.460 | -24.460 | unsolved |
| progress_reward + stable_landing_reward | 4 | -37.290 | -47.110 | unsolved |
| angle_penalty + potential_diff + success_bonus | 1 | -112.790 | -112.790 | unsolved |
| angle_penalty + landing_reward + potential_diff | 1 | -150.570 | -150.570 | unsolved |

## Previous interventions

- iter 2 (score=-43.990, structure=progress_reward + stable_landing_reward): selected_level:: Level 2，因为证据直接否定了当前组件的数学形态（乘积塌缩），且必要着陆引导职责缺失。 | selected_intervention:: 将 stable_landing_reward 从乘积形式改为加性组合，具体包括：
- iter 6 (score=-112.790, structure=angle_penalty + potential_diff + success_bonus): 4. `selected_level`: Level 2，触发条件为缺失必要职责（完成信号），且当前过程组件已能引导agent到达目标附近，需要将代理目标与任务完成对齐（proxy_to_completion_alignment）。 | 5. `selected_intervention`: 新增一个稀疏完成奖励组件`success_bonus` —— 当检测到双脚接触且姿态、位置、速度均满足软着陆条件时给予固定正奖励；其他所有组件保持不变。
- iter 7 (score=0.620, structure=angle_penalty + landing_quality_reward + potential_diff): 4. `selected_level`：Level 2，触发条件为sparse_to_dense——上一轮稀疏bonus失败，且任务需要基于接触、位置、速度、角度的连续完成证据。 | 5. `selected_intervention`：移除success_bonus，新增landing_quality_reward（双脚接触时对位置、速度、角度按乘积阈值输出连续值，系数5.0）。
- iter 8 (score=-150.570, structure=angle_penalty + landing_reward + potential_diff): 4. `selected_level`：Level 2 — the evidence pattern “task event almost never triggers, local feedback missing” matches `sparse_to_dense`; a structural change from a multiplicative sparse bonus to a continuous, additive pr | 5. `selected_intervention`：Change the `landing_quality_reward` component from a product of hard thresholds to a dense sum of independent proximity measures (position, velocity, angle, contact) using bounded linear decays
- iter 9 (score=-16.610, structure=angle_penalty + landing_quality_reward + potential_diff): 4. `selected_level`: Level 2 —— 证据模式为“proxy提高但外部任务不升”，触发proxy_to_completion_alignment变换。 | 5. `selected_intervention`: 将landing_reward从稠密加权和的proximity形态恢复为稀疏乘积形态(与best相同)，仅在有双脚接触时才能获得正奖励，并将系数k_landing由5.0提升至20.0以放大正确行为信号。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
