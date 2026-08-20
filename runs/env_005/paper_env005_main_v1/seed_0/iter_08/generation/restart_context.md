# Fresh Restart Evidence

- target_score: 6000.000
- best_score_so_far: 12.310

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| angular_velocity_penalty + forward_velocity_reward + height_deviation_penalty + lateral_velocity_penalty + orientation_reward | 1 | 12.310 | 12.310 | unsolved |
| angular_velocity_penalty + forward_velocity_reward + lateral_velocity_penalty + orientation_reward | 1 | 11.050 | 11.050 | unsolved |
| forward_velocity_reward + orientation_reward + stability_penalty | 3 | -49.530 | -49.530 | unsolved |
| forward_velocity_reward + stability_penalty | 1 | -123.550 | -123.550 | unsolved |
| angular_velocity_penalty + forward_velocity_reward + height_boundary_penalty + lateral_velocity_penalty + orientation_reward | 1 | -131.920 | -131.920 | unsolved |

## Previous interventions

- iter 2 (score=-50.900, structure=forward_velocity_reward + orientation_reward + stability_penalty): selected_level**: Level 2 — 必要信号缺失（身体倾斜反馈不可达），仅调系数无法解决17/20崩塌问题；干预类型为添加连续姿态奖励组件（sparse_to_dense变体：将隐式高度终止风险转化为显式倾斜反馈）。 | selected_intervention**: 新增`orientation_reward`组件，利用body quaternion计算身体z轴在世界坐标系中的投影`body_up_z`，奖励保持直立姿态。
- iter 3 (score=-137.450, structure=forward_velocity_reward + orientation_reward + stability_penalty): `selected_level`: Level 1 — the stability_penalty component has correct structure (hinge loss with appropriate margins for the healthy range), but its height sub-coefficient w_h=10 is too small relative to forward_veloci | `selected_intervention`: increase w_h from 10.0 to 50.0, a 5× amplification of the height penalty within the stability_penalty component. No other coefficients, margins, or mathematical forms are changed.
- iter 4 (score=-49.530, structure=forward_velocity_reward + orientation_reward + stability_penalty): `selected_level`: Level 2 — `global_to_local_gate`。触发条件：(1) forward_velocity_reward 是唯一主信号但其不顾姿态始终全额发放，证据为18/20失败终止时 forward 仍贡献42.5% signed_share；(2) 纯系数调整(w_h 10→50)已证实无效且恶化；(3) 需要结构变换将前进奖励门控在稳定状态上。 | `selected_intervention`: 唯一修改 forward_velocity_reward 组件，将其从 `w_fwd * body_x_vel` 变换为 `w_fwd * body_x_vel * stability_gate`，其中 stability_gate 由身体直立度与高度健康度连续乘积构成；其他组件保持 best 代码不变（w_h=10.0）。
- iter 6 (score=12.310, structure=angular_velocity_penalty + forward_velocity_reward + height_deviation_penalty + lateral_velocity_penalty + orientation_reward): 4. `selected_level`: Level 2** — a necessary signal (height management) is entirely absent, and the termination evidence (19/20 height-violation failures) directly supports adding it. This is the `unbounded_to_bounded` p | 5. `selected_intervention`

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
