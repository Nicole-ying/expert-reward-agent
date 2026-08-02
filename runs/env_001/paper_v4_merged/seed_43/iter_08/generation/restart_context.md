# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -87.190

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| action_cost + landing_contact_reward + landing_speed_gate + progress_shaping + shaped_progress | 1 | -87.190 | -87.190 | unsolved |
| action_cost + landing_contact_reward + progress_shaping + shaped_progress | 1 | -87.190 | -87.190 | unsolved |
| action_cost + angle_hinge_penalty + landing_contact_reward + progress_shaping + shaped_progress | 2 | -105.530 | -105.530 | unsolved |
| action_cost + angle_hinge + danger_penalty + progress_shaping | 1 | -117.480 | -117.480 | unsolved |
| action_cost + angle_hinge + progress_shaping | 1 | -117.880 | -117.880 | unsolved |
| action_cost + angle_hinge + landing_contact_reward + progress_shaping | 1 | -122.170 | -122.170 | unsolved |

## Previous interventions

- iter 2 (score=-117.480, structure=action_cost + angle_hinge + danger_penalty + progress_shaping): 4. `selected_level`：Level 2 结构变换——基于信号缺口与几乎死亡组件的证据，新增使用未利用观测的危险惩罚组件。 | 5. `selected_intervention`：新增 `danger_penalty` 组件，检测 `abs(nx)>1.2`、`ny<-0.2`、`abs(nangle)>0.8`、或速度幅值 >5.0 等致命状态，每命中步给予 −1.0 惩罚。
- iter 3 (score=-122.170, structure=action_cost + angle_hinge + landing_contact_reward + progress_shaping): selected_level：Level 2 — structural transform，因前轮迭代得分停滞且僵尸组件（danger_penalty active_rate=0%）未实现设计意图，需移除并替换为新职责信号。 | selected_intervention：删除danger_penalty，新增landing_contact_reward组件，基于支撑脚接触和到目标距离的连续bounded factor，以提供着陆指向性奖励。
- iter 4 (score=-87.190, structure=action_cost + landing_contact_reward + landing_speed_gate + progress_shaping + shaped_progress): 4. selected_level: Level 2 — structure change: remove zombie angle_hinge and replace with a landing_speed_gate that scales progress_shaping based on speed when close to target. | 5. selected_intervention: Delete angle_hinge; add `landing_speed_gate = 1.0 / (1.0 + 5.0 * speed_next * max(0.0, 1.0 - dist_next / 0.5))` and multiply progress_shaping by it. This one-component swap leaves action_cost an
- iter 5 (score=-87.190, structure=action_cost + landing_contact_reward + progress_shaping + shaped_progress): 4. selected_level: Level 2 – structural change, because the landing_speed_gate component is active 100% of the time and contributes ~100% signed share as a non‑reward artefact, requiring removal from the component output | 5. selected_intervention: Remove `landing_speed_gate` from the returned components dictionary; keep its computation and multiplication intact (used for shaping) but stop emitting it as a reward term.
- iter 6 (score=-114.350, structure=action_cost + angle_hinge_penalty + landing_contact_reward + progress_shaping + shaped_progress): 4. `selected_level`：Level 2 — 信号覆盖存在缺失（角度约束），需要添加一个新组件，属于结构变换。 | 5. `selected_intervention`：新增`angle_hinge_penalty`组件，对机身角度的绝对值超过0.3 rad的部分施加线性惩罚，系数0.03，引导飞行器保持水平姿态，避免触地坠毁。
- iter 7 (score=-105.530, structure=action_cost + angle_hinge_penalty + landing_contact_reward + progress_shaping + shaped_progress): 4. `selected_level`：Level 2 — 结构变换，触发条件：无界→有界（progress_shaping的负分支在坠毁时爆炸，需bounding）。 | 5. `selected_intervention`：仅修改progress_shaping组件，从potential-based无界差分变为基于距离增量的bounded improvement（进步系数0.5，退步系数0.05），以压制退步时的灾难性惩罚。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
