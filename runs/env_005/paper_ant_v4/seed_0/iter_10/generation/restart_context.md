# Fresh Restart Evidence

- target_score: 2000.000
- best_score_so_far: 203.900

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| ang_penalty + forward_reward + height_penalty + lat_penalty + upright_penalty | 3 | 203.900 | 52.690 | unsolved |
| angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward | 4 | 26.070 | 26.070 | unsolved |
| ang_penalty + forward_reward + lat_penalty | 2 | -48.960 | -172.450 | unsolved |

## Previous interventions

- iter 3 (score=24.080, structure=angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward): 4. **selected_level**：Level 2，因为必要信号可达但现有门控阈值（0.35）未能及时衰减速度奖励，属于“约束在无关阶段妨碍探索”变体——需要在高度下降初期就施加阶段相关的速度抑制，故选择global_to_local_gate思路修改velocity_reward中的门控。 | 5. **selected_intervention**：仅修改velocity_reward组件内的health_factor低高度阈值，将线性衰减区间从[0.2, 0.35]扩大到[0.2, 0.5]，使身体高度低于0.5时前向速度奖励即开始成比例衰减。
- iter 4 (score=26.070, structure=angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward): 4. **selected_level**: Level 1 — the height_penalty component has a reasonable mathematical form (quadratic, signed), but its extreme coefficient (10) makes it too strong relative to other components, poisoning the rewar | 5. **selected_intervention**: Reduce the height_penalty coefficient from 10.0 to 2.0, keeping the quadratic form and all other components unchanged.
- iter 6 (score=-172.450, structure=ang_penalty + forward_reward + lat_penalty): 4. selected_level：Level 2（结构变换），将奖励计算从 obs 切换到 next_obs，以纠正信用分配错误；这符合“修正观测源”这类最小但可验证的数学结构修改。 | 5. selected_intervention：唯一目标组件 forward_reward。修改其计算：用 next_obs 替换 obs 来获取 body_x_vel 以及计算 combined_gate（高度和直立），其他结构、系数和惩罚项保持不变。
- iter 7 (score=203.900, structure=ang_penalty + forward_reward + height_penalty + lat_penalty + upright_penalty): 4. selected_level：Level 3 重建主骨架——当前骨架族（forward_reward + lat/ang penalty + 门控）连续两轮未刷新best、历史最佳仅26（远低于target 2000的25%），且已做Level 1尺度调整无效，满足重建条件。 | 5. selected_intervention：以iter4的组分框架为基础，将前进奖励从线性改为平方项（仅对正前向速度）以提供更强梯度，移除乘性健康/直立门控，改用温和的加性高度/直立惩罚维持存活，保留侧向与角速度惩罚。
- iter 8 (score=147.710, structure=ang_penalty + forward_reward + height_penalty + lat_penalty + upright_penalty): 4. **selected_level**：Level 2，因为直立惩罚的数学形态导致其只在倾斜≥60°后起作用，证据表明agent在该阶段已难以恢复，需要连续惩罚来提供更密集的直立梯度。 | 5. **selected_intervention**：仅修改`upright_penalty`，从`10.0 * max(0.0, 0.5 - body_up_z)`变为`5.0 * ((1.0 - body_up_z) ** 2)`，使任何偏离完全直立的状态都产生连续惩罚，强度随倾角平方增长。
- iter 9 (score=52.690, structure=ang_penalty + forward_reward + height_penalty + lat_penalty + upright_penalty): 4. `selected_level`：Level 2，因为同一骨架已连续两轮未刷新best（iter 7→8下降），且目标完成率仅7.4%；高度惩罚尺度太弱，改变其数学形态以形成有效保护。 | 5. `selected_intervention`：唯一目标组件为`height_penalty`，将低高度惩罚由线性改为二次且阈值从0.3提前到0.4，系数从20提升到100，使其在接近摔倒时急剧增大。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
