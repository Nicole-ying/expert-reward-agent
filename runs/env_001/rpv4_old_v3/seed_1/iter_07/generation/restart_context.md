# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -108.180

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| goal_approach + landing_comfort + soft_landing | 1 | -108.180 | -108.180 | unsolved |
| crash_avoidance + goal_approach + soft_landing | 2 | -109.580 | -109.580 | unsolved |
| contact_reward + goal_approach + soft_landing | 1 | -111.060 | -111.060 | unsolved |
| goal_approach + landing_success + soft_landing | 2 | -115.210 | -119.000 | unsolved |

## Previous interventions

- iter 2 (score=-109.580, structure=crash_avoidance + goal_approach + soft_landing): 4. `selected_level`：Level 2（修改组件数学形态），因为 soft_landing 的触发条件门控导致其死组件，需改为无门控的持续惩罚以提供可学习的速度控制信号。 | 5. `selected_intervention`：唯一目标组件为 soft_landing，将其从“低空门控二次惩罚”改为“全高度线性惩罚”，同时纳入姿态角度和角速度，使其在全程产生负向激励，引导 agent 学习减速与稳定姿态。
- iter 3 (score=-115.210, structure=goal_approach + landing_success + soft_landing): 4. `selected_level`：Level 2（修改公式形式）——crash_avoidance 为死组件，替换其函数形式为着陆成功奖励是最小且可验证的干预。 | 5. `selected_intervention`：唯一目标组件为 `crash_avoidance`，将其替换为 `landing_success`：当双腿接触且满足低速、直立、低角速度条件时，给予一次性正向大奖励。
- iter 4 (score=-111.060, structure=contact_reward + goal_approach + soft_landing): 4. `selected_level`：Level 2，将死组件 landing_success 的稀疏奖励改为连续的双腿接触奖励，使其在接触的每一步都产生正向激励，不再依赖稳定条件的罕见同时满足。 | 5. `selected_intervention`：唯一目标组件为 landing_success，将其替换为 contact_reward（每步当 left_contact_next>0.5 且 right_contact_next>0.5 时给予 +1.0），并移除 crash_avoidance。
- iter 5 (score=-119.000, structure=goal_approach + landing_success + soft_landing): 4. selected_level：Level 2—contact_reward为死组件（active_rate<2%），其数学形态（每步+1仅双腿接触）无法激励稳定着陆，需将稀疏的接触奖励改为一次性成功条件奖励。 | 5. selected_intervention：唯一目标组件contact_reward，将其替换为landing_success，当同时满足双腿接触、低速、直立、低角速度条件时给予一次性+100奖励。
- iter 6 (score=-108.180, structure=goal_approach + landing_comfort + soft_landing): 4. `selected_level`：Level 2，因landing_success活性极低，数学形式无效，需将稀疏bonus改为连续舒适奖励，基于历史最佳v2骨架替换一个未触发组件。 | 5. `selected_intervention`：基于v2代码，把crash_avoidance（几乎不触发）替换为landing_comfort，该组件在双腿接触时根据速度和姿态误差给出连续奖励，最大每步1.0，从而提供逐步改善的梯度。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
