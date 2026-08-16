# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: -103.790

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| action_penalty + landing_reward + progress_reward | 2 | -103.790 | -103.790 | unsolved |
| action_penalty + contact_reward + descent_reward + progress_reward | 1 | -120.100 | -120.100 | unsolved |
| contact_reward + orientation_penalty + progress_reward + vel_damping | 4 | -124.550 | -159.200 | unsolved |

## Previous interventions

- iter 2 (score=-168.020, structure=contact_reward + orientation_penalty + progress_reward + vel_damping): selected_level**：L2: 这里不是单纯的权重问题，vel_damping 的数学条件需要修改才能从“死组件”变为有效信号。因此选择 L2 修改其公式形态。 | selected_intervention: 目标组件：`vel_damping`（将其重定义为“接近地面时的速度惩罚”）
- iter 3 (score=-124.550, structure=contact_reward + orientation_penalty + progress_reward + vel_damping): selected_level**：**L2**（改动数学形态） | selected_intervention**：
- iter 6 (score=-103.790, structure=action_penalty + landing_reward + progress_reward): 历史记录中（通过 `read_past_reflection` 和 `read_reward_code` 检索）**未出现**“将 landing 奖励改为连续的高度‑速度‑姿态质量函数”的尝试，因此本轮修改不是已有语义的重复。 | selected_level: L2** — 改变 landing_reward 的数学形态，从离散双腿接触奖励变为连续的高度/速度/姿态质量函数。历史中未做过此变换，且当前分数离目标（200）差距远超 10%，L1 微调不足以解决稀疏性问题。 | selected_intervention: 将 `landing_reward` 改写为：
- iter 7 (score=-120.100, structure=action_penalty + contact_reward + descent_reward + progress_reward): selected_level**：**L2: 将 `landing_reward` 从“连续状态质量奖励”改为“下降过程奖励 + 接触事件奖励”，从根本上消除静态悬停可利用点。同步删除 `progress_reward` 的健康门控使其重新生效。 | selected_intervention**：

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
