# 诊断与决策

## 1. evidence
Subagent 报告明确指出：contact_landing_reward 的 per-step 量级（0.588）是 goal_approach_progress（0.016）的 **37 倍**，导致接触信号压制航行引导。历史最佳 iter2（142.32 分）使用几何平均质量门槛 + 相同 10.0 approach 系数，当前 iter4（103.63 分）改回加权和后 contact 激活率降至 13.8%，但 approach 同步崩塌 92%（159→12 episode sum），证明非质量门槛本身的问题，而是 approach 系数结构性不足。

## 2. behavior_diagnosis
agent 学会"磨蹭到目标附近才尝试接触"：13/20 终止（高完成率），但回合长 737 步且得分低于历史最佳。contact 奖励在末尾集中爆发（13.8% 激活率占 90.7% 份额），approach 虽全程激活（99.3%）但单步贡献仅 0.016，无法驱动早期高效航行。

## 3. signal_completeness
职责完备：approach 提供航行引导，contact 提供着陆完成信号，velocity/orientation 提供安全约束。但 approach 量级不足导致"可达性"失效——梯度被 contact 淹没，agent 无法在早期阶段学会高效接近。

## 4. selected_level
**Level 1：系数调整**。Subagent 明确推荐将 goal_approach_progress 系数从 10.0 提升至 50-100 量级。证据充分：
- iter2 的几何平均门槛配置下 approach 仍能贡献健康信号（0.027 per-step）
- iter3→4 改数学形态后 approach 崩塌不是因为质量门槛本身，而是 contact 基础系数（5.0）在加权和下放大了量级差距
- 当前 contact per-step 0.588 vs approach 0.016 的 37 倍差距需要系数层级纠正

## 5. selected_intervention
**唯一目标组件：goal_approach_progress**。将系数从 10.0 提升至 **80.0**（位于 Subagent 推荐 50-100 区间中点），保持其他三个组件（velocity_brake_near_target、orientation_stability、contact_landing_reward）完全不变。基于 iter2 最佳代码（几何平均质量门槛）进行修改，因其 contact 配置已被证明有效（142.32 分），只需补强 approach 驱动力。

## 6. falsifiable_hypothesis
提升 approach 系数至 80.0 后，预期 per-step 贡献从 0.016 升至约 0.128（8 倍），与 contact 的 0.588 形成 1:4.6 比例。这将在早期阶段（距离大、接触未发生）让 approach 主导梯度，agent 应学会更直接的航行轨迹，减少徘徊时间。

## 7. expected_next_round
- **episode_length**：从 737 降至 500-600（更高效接近路径）
- **goal_approach_progress episode_sum**：从 12 升至 80-120（全程累积更多距离改善）
- **goal_approach_progress active_rate**：保持 99%+（密集引导不受影响）
- **contact_landing_reward episode_sum**：保持 400-500 范围（质量门槛配置已验证有效）
- **score**：目标超越 142.32，预期 160-180（更高效航行 + 保持高质量接触）

## 8. main_risk
approach 系数过强可能在接近阶段压制 velocity_brake 和 orientation_stability 的安全约束，导致高速撞击。但当前 velocity_brake 使用 proximity_factor 动态缩放（近距离时惩罚放大），应能抵消该风险。若出现高速撞击，下一轮需提升 velocity_brake 基础系数从 -2.0 至 -5.0。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation and soft landing task.
    
    Task: Navigate from random perturbed initial position to target landing pad,
    achieve stable low-speed contact with upright orientation.
    
    Components:
    - goal_approach_progress: dense shaping via distance improvement delta (COEFFICIENT BOOSTED)
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
    components["goal_approach_progress"] = 80.0 * distance_delta
    
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