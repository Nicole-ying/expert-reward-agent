# 奖励设计技法与失败模式库

每一条 = 独立知识单元。检索命中其中一条即可指导修订。

## penalty_magnitude_control
- 类型：设计技法
- 症状：最终策略中某个持续惩罚的绝对累计量明显压过同时间语义的主学习信号，且 external score 差、agent 可能不动。历史经验中，dense penalty 的 `ratio_to_progress` 绝对值超过约 0.5 是值得优先检查的预警线，不是通用判定线。
- 原理：惩罚信号主导时，agent 为避免惩罚而选择不行动。惩罚应是方向盘，不是刹车。
- 修复：降低该惩罚系数（至少10倍），或改为距离门控（只在目标附近生效），或用平滑 bounded 函数代替线性惩罚。
- 关键词：penalty dominating, progress weak, agent stuck, conservative, afraid to move

## bounded_continuous_proxy
- 类型：设计技法
- 症状：完成信号（如 landing bonus）的 `active_rate`（旧日志名 `nonzero_rate`）接近 0；历史经验中低于约 1%~2% 通常意味着策略几乎得不到任务相关反馈，但阈值需结合事件本身频率判断。
- 原理：二值条件（if near and slow and upright: bonus=2）无梯度。连续乘积 near_factor * slow_factor * upright_factor 提供密集信号。1/(1+kx) 或 max(0,1-x/D) 自动 bounded 防数值爆炸。
- 修复：用连续乘积因子取代 if 条件；每个因子用 bounded 形式 max(0, 1-dimension/D_threshold)。或使用 proximity=1/(1+5*dist)。
- 关键词：sparse reward, binary condition, low trigger rate, continuous shaping, product shaping, bounded function

## distance_gated_constraint
- 类型：设计技法
- 症状：远离目标时惩罚阻碍 agent 靠近，episode 长但无进展。
- 原理：远处需要大动作，不应施加精细约束。用距离门控让惩罚只在目标区生效。
- 修复：penalty *= max(0, 1 - dist/gate_radius)。gate_radius 取初始距离的 30%~50%。
- 关键词：early exploration, local penalty, gated constraint, distance conditional

## anti_oscillation_potential
- 类型：设计技法
- 症状：progress_delta 在目标附近震荡，episode 长但不完成，landing proxy 触发率低。
- 原理：delta distance 无最优策略不变性保证。势能塑形是唯一理论正确的塑形方式。bounded proximity 天然防震荡。
- 修复：用 bounded_proximity=1/(1+5*dist) 或 clip progress_delta 到 [-0.5, 0.5]；或引入势能塑形 F=γΦ(s')-Φ(s) 用 Φ=-distance。
- 关键词：oscillation, goal near, potential shaping, clipping, proximity, hovering

## stage_weighted_reward
- 类型：设计技法
- 症状：训练早期需要探索，后期需要精细控制，固定权重难以兼顾。
- 原理：利用 training_progress 动态调整 exploration vs precision 权重。early 鼓励运动，late 追求精度。
- 修复：early_weight=1-t, mid_weight=4t(1-t), late_weight=t。对 progress 用 early_weight，对 stability 用 late_weight。
- 关键词：curriculum, training progress, dynamic weight, exploration, phase-based

## coefficient_scale_awareness
- 类型：设计技法
- 症状：所有组件的 mean 值都在 0.001 量级或都在 100 量级，信号缺乏区分度。
- 原理：不同组件需要不同量级来体现优先级。progress 应该是最强的正信号，惩罚应该是弱而常在的背景。
- 修复：先比较同时间语义组件的 episode_abs_sum_mean、magnitude_share 与 active_rate；若主学习信号持续被约束项压制，单独降低该约束系数并在下一轮验证。对“dense progress + dense penalty”的早期学习阶段，可把学习信号约为惩罚的 10 倍（即 penalty/progress 约 0.1）作为保守起点；这是v6中的有效经验，不适用于事件 bonus、持续状态奖励或正负抵消严重的差分均值。
- 关键词：scale, magnitude, ratio, coefficient tuning, balance

## stability_penalty_dominance
- 类型：失败模式
- 症状：stability_penalty 在最终策略奖励构成中持续占主导，external 低、episode 短，且其数学形态本身尚未被证伪。对可比的dense信号，`|ratio_to_progress| > 0.5`可作为优先检查并尝试降系数的经验触发器。
- 风险：agent 保守，不敢移动，或快速坠毁。
- 修复：降低惩罚系数（至少10倍）或使用距离门控。关联技法 penalty_magnitude_control。
- 关键词：stability penalty, conservative agent, penalty dominating, crash

## goal_near_oscillation
- 类型：失败模式
- 症状：progress 正常，episode 长（>800），landing proxy 触发率 < 10%。
- 风险：agent 在目标附近徘徊不降落。
- 修复：添加连续 low-speed+low-angle+proximity 乘积 shaping。关联技法 bounded_continuous_proxy, anti_oscillation_potential。
- 关键词：hovering, oscillation, near goal, landing proxy sparse, long episode

## contact_reward_hacking
- 类型：奖励黑客
- 症状：contact 相关奖励高，但 external score 低。
- 风险：agent 利用接触信号而非完成任务。
- 修复：contact 奖励只能在 near target + low speed + stable angle + both supports 下触发。关联技法 bounded_continuous_proxy。
- 关键词：contact hacking, exploit, cheating, false positive

## early_failure_or_crash
- 类型：失败模式
- 症状：external score 极负（<-100），episode length 很短（<150），提前终止比例 > 50%。
- 风险：agent 快速坠毁，奖励未引导安全行为。
- 修复：增加密集的 safe altitude/approach 信号；大幅降低稳定性惩罚。关联技法 bounded_continuous_proxy, penalty_magnitude_control。
- 关键词：early crash, short episode, negative external, termination

## high_reward_without_success
- 类型：奖励黑客
- 症状：generated_reward 高（>0），但 external score 差（<-50）。
- 风险：agent 优化了自定义奖励而非真实任务。
- 修复：重新检查各组件与 external 相关性，移除可钻空子的项。关联技法 penalty_magnitude_control。
- 关键词：reward hacking, misalignment, high custom reward low external, exploit

## sparse_completion_proxy
- 类型：失败模式
- 症状：completion/landing bonus 的 `active_rate`（旧名 `nonzero_rate`）低于约1%时，通常无法提供可达的学习引导；天然极低频的一次性事件需结合每回合触发次数解释。
- 风险：agent 收不到完成任务的信号，无法学习最终步骤。
- 修复：用连续 bounded 函数取代硬性二值奖励。关联技法 bounded_continuous_proxy。
- 关键词：sparse reward, completion proxy, low trigger, binary bonus
# Locomotion-specific diagnosis cards

## fast_then_fail
- 症状：前进相关组件较强，但外部得分不高、episode明显偏短，策略以高速或失稳动作换取短期前进。
- 原理：状态速度奖励可能鼓励占据高速状态，却没有表达持续通过地形的要求。
- 修复：先检查稳定职责是否缺失；按证据选择轻量姿态约束、局部门控，或把状态速度改为兼顾持续性的有界结构。不要仅因速度高就直接增加全局重罚。
- 关键词：locomotion, fast then fail, forward reward, early fall, unstable gait

## hover_or_stand_still
- 症状：episode较长但外部得分低，alive/stability类正向贡献持续存在，前进信号弱或被动作代价压制。
- 原理：生存和低动作成本可以形成原地不动的局部最优。
- 修复：削弱无条件存活收益或动作代价，确保可达的前进职责是主要任务驱动；下一轮同时验证episode长度与外部得分。
- 关键词：locomotion, stand still, alive bonus, action penalty, conservative policy

## locomotion_energy_domination
- 症状：能耗或动作惩罚绝对累计贡献过大，策略动作幅度很小且前进不足。
- 原理：完成任务前过早优化控制成本会抑制步态探索。
- 修复：Level 1降低单一动作代价；若仍失败，Level 2将代价门控到已具备前进能力的阶段。事件、状态值和逐步差分不能套用同一固定ratio阈值。
- 关键词：locomotion, energy penalty, torque cost, afraid to move
