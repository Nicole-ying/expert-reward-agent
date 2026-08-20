# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -46.510

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| action_cost + angle_penalty + progress_shaping + success_landing_bonus | 5 | -46.510 | -98.930 | unsolved |
| action_cost + angle_penalty + landing_velocity_penalty + progress_shaping | 1 | -155.980 | -155.980 | unsolved |
| action_cost + angle_penalty + crash_penalty + progress_shaping + success_landing_bonus | 1 | -203.450 | -203.450 | unsolved |

## Previous interventions

- iter 2 (score=-119.510, structure=action_cost + angle_penalty + progress_shaping + success_landing_bonus): 4. `selected_level`：Level 2——着陆速度惩罚组件形式错误（稀疏惩罚且门控过窄，形同虚设），需替换为成功着陆奖励，即从惩罚形态变为正向bonus。 | 5. `selected_intervention`：将`landing_velocity_penalty`组件整体替换为`success_landing_bonus`，基于next_obs的接触、位置、速度、角度给出稀疏大正奖励。
- iter 3 (score=-203.450, structure=action_cost + angle_penalty + crash_penalty + progress_shaping + success_landing_bonus): 4. selected_level：Level 1（系数/惩罚调整）。progress_shaping 形态本身合理，但需要补充 crash 惩罚以阻断 crash 路径，引导 agent 尝试更安全的着陆。 | 5. selected_intervention：新增 `crash_penalty` 组件，在接触地面但未满足软着陆条件、或 x 位置超出视区边界时给予 -100，与 success_landing_bonus 互斥，其余组件不变。
- iter 4 (score=-114.380, structure=action_cost + angle_penalty + progress_shaping + success_landing_bonus): 4. `selected_level`：Level 2，修改 success_landing_bonus 的触发条件，放宽阈值，同时移除 crash_penalty 以恢复 v2 有效结构。 | 5. `selected_intervention`：删除 crash_penalty，将 success_landing_bonus 的判定距离阈值放宽到 0.2，速度阈值放宽到 0.5，角度阈值放宽到 0.2。
- iter 5 (score=-46.510, structure=action_cost + angle_penalty + progress_shaping + success_landing_bonus): 4. `selected_level`：Level 2，基于v4成功经验进一步放宽success_landing_bonus的触发阈值以提升active_rate，不做框架级重建。 | 5. `selected_intervention`：将success_landing_bonus的判定条件中的距离阈值由0.2提至0.4，水平/垂直速度阈值由0.5提至1.0，角度阈值由0.2提至0.5，保持接触条件不变。
- iter 6 (score=-111.180, structure=action_cost + angle_penalty + progress_shaping + success_landing_bonus): 4. `selected_level`：Level 2，success_landing_bonus的数学形态传达了错误激励（状态值可连续触发），需改为事件触发并收紧阈值。 | 5. `selected_intervention`：将success_landing_bonus修改为仅在从无接触转为有接触时触发，且收紧软着陆条件（距离<0.3，速度绝对值<0.3，角度绝对值<0.3），奖金提高至200。
- iter 7 (score=-98.930, structure=action_cost + angle_penalty + progress_shaping + success_landing_bonus): 4. `selected_level`：Level 2——修改 success_landing_bonus 的数学形态，从每步状态值奖励改为基于接触转变的一次性事件奖励，使用 v5 的宽松阈值以避免死组件，同时解决滞留。 | 5. `selected_intervention`：将 success_landing_bonus 改为在支撑脚从无接触到有接触的瞬间，若满足 v5 的软着陆条件（dist<0.4, vel_abs<1.0, angle_abs<0.5）则给予 +100，其余时刻为 0。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
