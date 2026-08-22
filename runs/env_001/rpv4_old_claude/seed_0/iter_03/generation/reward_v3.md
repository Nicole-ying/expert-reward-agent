# 诊断与干预决策

## 1. evidence
当前 iter2 score=142.32 已超过历史最差（iter1 -82.86）225 分，terminated 从 1/20 跃升至 16/20，len 从 984.9 压缩至 721.1，说明骨架级修改（删除稀疏 terminal_soft_landing_bonus，新增密集质量门控 contact_landing_reward）驱动了策略从"漂浮拖时间"到"主动接近并着陆"的行为跃迁。距离 target_score=200 还差 57.7 分（达成率 71.2%），4/20 回合仍 truncated（未触发终止），说明仍有提升空间。

Subagent 诊断明确指出：goal_approach_progress 覆盖率 98.8%（健康），但 per-step 均值仅 0.017，比 velocity_brake 的 -0.036 小一半，比 contact_landing_reward 的 per-active-step（≈4.15）低约 240 倍——是"淹没型"信号。当前未暴露问题是因为 contact_landing_reward 的 proximity_quality 门控间接兜底，但这是脆弱耦合，若后续调参改变门控逻辑，早期接近阶段可能失去独立引导。

## 2. behavior_diagnosis
策略已学会"接近 → 触地 → 维持接触"完整链路，16/20 回合成功触发终止，说明 contact_landing_reward 的质量门控有效对齐了"低速+正姿态+近目标"的联合约束。4/20 truncated 回合可能来自两个原因：(1) 早期接近阶段效率不足，goal_approach_progress 作为主引导信号被淹没，导致部分回合在远距离徘徊耗尽步数；(2) 接触阶段的质量门控虽然方向正确，但 contact_landing_reward 系数（5.0）可能相对偏低，导致部分回合在接近目标后未能及时学会"快速稳定接触"。

## 3. signal_completeness
当前骨架已包含必要职责：
- 任务进展引导：goal_approach_progress（距离改善差分）
- 安全约束：velocity_brake_near_target（速度惩罚）+ orientation_stability（姿态惩罚）
- 完成信号：contact_landing_reward（接触奖励 + 质量门控）

职责完备，但 goal_approach_progress 量级被压制，无法独立发挥早期引导作用。

## 4. selected_level
**Level 1：系数调整**

触发条件：goal_approach_progress 的数学形态（距离差分 Δd）和覆盖率（98.8%）均正确，问题仅在于量级失衡——per-step 0.017 被 contact_landing_reward 的 per-active-step 4.15 压制约 240 倍，导致其作为主引导信号的作用被淹没。Subagent 明确推荐"评估是否需要提高其系数"，当前 coeff=10.0，需提升至与 contact 信号在数值尺度上可比的量级（不追求完全对等，但至少要让接近阶段的 per-step 贡献与接触阶段的平均贡献在同一数量级）。

## 5. selected_intervention
**唯一目标组件：goal_approach_progress**

具体修改：系数从 10.0 提升至 50.0（5 倍放大）。

理由：
- 当前 per-step 0.017 × 5 = 0.085，将接近 velocity_brake 的 per-step 量级（-0.036 的绝对值），使其在早期飞行阶段能独立起到有效引导作用，不再完全依赖 contact_landing_reward 的 proximity_quality 兜底。
- contact_landing_reward 的 per-active-step ≈4.15，假设单个接触阶段持续 100 步（14.6% active_rate 对应约 105 步），累积贡献约 415；goal_approach_progress 在 700 步飞行阶段，若每步均值从 0.017 提升至 0.085，累积贡献约 59.5，与接触阶段的主导地位相比仍是辅助角色（约 1:7），但足以在接近阶段提供清晰的梯度信号。
- 5 倍放大是保守选择：避免过度放大导致 agent 只追求接近而不学习接触质量门控（若放大 10 倍以上，可能压制 contact_landing_reward 的主导地位）。

## 6. falsifiable_hypothesis
提升 goal_approach_progress 系数后，策略在早期飞行阶段将获得更强的"接近目标"梯度信号，预期效果：
- 4/20 truncated 回合中，若部分来自"远距离徘徊耗尽步数"，则下一轮 truncated 比例应下降（terminated 比例上升）；
- 平均 episode_length 可能略微缩短（更快收敛到目标区域）；
- score 应上升（更高效的接近 + 终止比例提升）。

可证伪性：若下一轮 score 未上升或 truncated 比例未下降，说明当前瓶颈不在早期引导量级，而可能在接触阶段的质量门控强度或其他约束。

## 7. expected_next_round
- **score**：从 142.32 上升至 160-180 区间（假设 4/20 truncated 中有 2-3 个转为 terminated）
- **episode_length**：从 721.1 小幅下降至 680-700（更高效的接近）
- **terminated**：从 16/20 上升至 18-19/20
- **goal_approach_progress per-step**：从 0.017 上升至 0.07-0.09
- **goal_approach_progress signed_share**：从 2.6% 上升至 8-12%（仍不压制 contact_landing_reward 主导地位）

## 8. main_risk
过度放大 goal_approach_progress 可能导致 agent 学会"快速接近但忽略减速/姿态准备"，即只追求距离缩短而不学习接触质量门控，最终虽然快速到达目标区域但以高速/错误姿态坠毁。当前 5 倍放大（10.0→50.0）是保守选择，预期不会完全压制 contact_landing_reward（其 per-active-step 仍约 50 倍于提升后的 goal_approach_progress per-step），但若下一轮出现 terminated 上升但 score 下降的情况（提前坠毁），则需回退并转而提升 contact_landing_reward 系数或调整其质量门控权重。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation and soft landing task.
    
    Task: Navigate from random perturbed initial position to target landing pad,
    achieve stable low-speed contact with upright orientation.
    
    Components:
    - goal_approach_progress: dense shaping via distance improvement delta (BOOSTED)
    - velocity_brake_near_target: quadratic velocity penalty scaled by proximity
    - orientation_stability: quadratic penalty on angle and angular velocity
    - contact_landing_reward: continuous contact reward modulated by quality factors
    """
    
    # Extract observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle, next_ang_vel = next_obs[4], next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    components = {}
    
    # === Component A: goal_approach_progress (main learning signal) ===
    # Use potential-based shaping: reward improvement in distance to target
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (next_x_pos**2 + next_y_pos**2) ** 0.5
    distance_delta = current_distance - next_distance  # positive when approaching
    components["goal_approach_progress"] = 50.0 * distance_delta
    
    # === Component B: velocity_brake_near_target (safety constraint) ===
    # Quadratic velocity penalty, scaled by proximity to encourage slow approach
    velocity_magnitude = (next_x_vel**2 + next_y_vel**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + next_distance)  # approaches 1 when close to target
    components["velocity_brake_near_target"] = -2.0 * proximity_factor * velocity_magnitude**2
    
    # === Component C: orientation_stability (safety constraint) ===
    # Quadratic penalty on angle deviation and angular velocity
    angle_penalty = next_angle**2
    ang_vel_penalty = next_ang_vel**2
    components["orientation_stability"] = -1.0 * (angle_penalty + 0.5 * ang_vel_penalty)
    
    # === Component D: contact_landing_reward (dense contact guidance) ===
    # Continuous reward for contact, modulated by landing quality factors
    # Base: any contact (left or right)
    contact_active = max(next_left_contact, next_right_contact)
    
    # Quality modulator 1: velocity (lower is better, bounded to [0,1])
    velocity_quality = max(0.0, 1.0 - velocity_magnitude / 1.0)
    
    # Quality modulator 2: orientation (smaller angle is better, bounded to [0,1])
    angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)
    
    # Quality modulator 3: proximity to target (closer is better, bounded to [0,1])
    proximity_quality = max(0.0, 1.0 - next_distance / 1.0)
    
    # Combined: contact reward scaled by quality factors
    # Using product to require joint satisfaction, but normalized to avoid collapse
    quality_factor = (velocity_quality * angle_quality * proximity_quality) ** 0.333
    components["contact_landing_reward"] = 5.0 * contact_active * quality_factor
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```