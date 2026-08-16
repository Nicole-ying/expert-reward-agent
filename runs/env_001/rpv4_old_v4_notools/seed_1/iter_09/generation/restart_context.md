# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: 39.440

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| attitude_penalty + proximity_progress + soft_landing_bonus | 4 | 39.440 | 39.440 | unsolved |
| attitude_penalty + landing_quality_reward + proximity_progress + speed_penalty + touchdown_reward | 1 | -24.390 | -24.390 | unsolved |
| attitude_penalty + leg_contact_reward + proximity_progress + soft_landing_bonus + speed_penalty | 1 | -39.690 | -39.690 | unsolved |
| attitude_penalty + proximity_progress + soft_landing_bonus + speed_penalty + touchdown_reward | 1 | -51.020 | -51.020 | unsolved |
| attitude_penalty + proximity_progress + soft_landing_bonus + speed_penalty | 1 | -108.890 | -108.890 | unsolved |

## Previous interventions

- iter 4 (score=39.440, structure=attitude_penalty + proximity_progress + soft_landing_bonus): selected_level**：L2: 不能仅靠 L1 调系数解决 exploit，必须改变 `soft_landing_bonus` 的数学形态，从持续过程奖励变为首次着陆事件的一次性奖励，并嵌入位置精度因子。 | selected_intervention: 目标组件 `soft_landing_bonus`：
- iter 5 (score=-108.890, structure=attitude_penalty + proximity_progress + soft_landing_bonus + speed_penalty): selected_level: L2。需要修改 `soft_landing_bonus` 的数学形态（从可重复触发的事件改为一次性成功事件），并联动新增一个速度惩罚组件（服务于同一语义——引导减速、杜绝弹跳刷分）。 | selected_intervention: 1. 将 `soft_landing_bonus` 重构为 **成功着陆一次性奖励**：仅在从“不满足成功条件”到“满足成功条件”的转变时给予 200 奖励。成功条件严格：双腿接触、位置足够近、速度足够低、姿态足够平。
- iter 6 (score=-39.690, structure=attitude_penalty + leg_contact_reward + proximity_progress + soft_landing_bonus + speed_penalty): selected_level**：L2 —— 需要引入一个新的连续奖励组件来填补信号空缺，使腿接触行为得到引导。 | selected_intervention**：
- iter 7 (score=-51.020, structure=attitude_penalty + proximity_progress + soft_landing_bonus + speed_penalty + touchdown_reward): selected_level: L2 – 修改 leg_contact_reward 的数学形态，将其替换为基于着陆质量的连续奖励，同时微调 proximity_progress 权重以平衡，整体服务于同一语义：为着陆阶段提供精细梯度信号。 | selected_intervention: 删除原 `leg_contact_reward`，新增 `touchdown_reward`：只在有腿接触时，以 `(1 - dist/0.3) * (1 - speed/0.3) * (1 - angle/0.2)` 刻画着落质量，系数 10.0，clip 到 [0, 1] 再乘系数，避免负值。
- iter 8 (score=-24.390, structure=attitude_penalty + landing_quality_reward + proximity_progress + speed_penalty + touchdown_reward): selected_level: L2** — 需要修改数学形态，而不仅仅是调参。稀疏事件奖励改为密集每步奖励、速度惩罚引入阶段门控、touchdown quality 从乘积改为加和，这些都不是单纯系数调整。 | selected_intervention: 1. **移除 `soft_landing_bonus`**，替换为 `landing_quality_reward`——成功条件满足时每步给 +10，持续奖励稳定着陆，消除大额脉冲。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
