# Response Record

# 诊断与干预决策

## 1. evidence
iter 2→3 唯一修改是将 goal_approach_progress 系数从 10.0 提升至 50.0，但实际效果是：contact_landing_reward 的 active_rate 从 50.8% 暴跌至 6.3%（降低 9 倍），回合长度增加 209 步，terminated 从 16/20 降至 7/20，score 下降 105.35。组件 per-step 量级对比显示：contact=0.26/step 远超 progress=0.06/step（4 倍），但 contact 的稀疏触发（仅 6.3% 时间激活）说明其质量门槛过严。

## 2. behavior_diagnosis
策略从 iter 2 的"稳定接触但未达终止条件"退化为"快速冲刺→短暂接触→弹开"。增强导航信号反而破坏了接触持久性，因为 progress 奖励"快速逼近"与 contact 要求的"缓慢稳定接触"存在根本冲突，且 contact 组件的三项质量因子乘积设计（velocity_quality × angle_quality × proximity_quality）在高速接近时坍缩为零。

## 3. signal_completeness
信号完备。goal_approach_progress 提供导航引导，velocity_brake_near_target + orientation_stability 提供约束，contact_landing_reward 提供接触引导。问题不在缺失职责，而在组件间的数学形态冲突。

## 4. selected_level
**Level 2：数学形态修正**

触发条件：contact_landing_reward 的 active_rate 从 50.8% 跌至 6.3% 不是量级问题，而是公式本身传达错误激励——三项质量因子的乘积设计要求同时满足低速、正姿态、近距离，任何一项偏离即导致奖励归零，与"快速接近"存在根本冲突。

## 5. selected_intervention
**目标组件**：contact_landing_reward

**具体修改**：将三项质量因子从乘积（要求联合满足）改为加权和（允许部分补偿）。理由：
- 当前 `quality_factor = (v_q × a_q × p_q)^0.333` 在任一维度为 0 时整体归零
- 改为 `quality_factor = 0.4×v_q + 0.3×a_q + 0.3×p_q` 允许"速度偏高但姿态和位置尚可"时仍获得部分奖励
- 权重分配：速度最重要（0.4），姿态和位置次之（各 0.3）

基于 iter 2（best）代码修改，不动其他组件。

## 6. falsifiable_hypothesis
改用加权和后，策略在"快速接近但姿态/位置尚可"阶段仍能获得中等 contact 奖励，鼓励"逐步刹车"而非"瞬间归零后弹开"。预期 contact 的 active_rate 从 6.3% 回升至 30%+ 区间，回合长度下降，terminated 比例提高。

## 7. expected_next_round
- contact_landing_reward 的 active_rate 应从 6.3% 回升至 25-40% 区间
- 回合长度应从 930 步降至 750-850 步
- terminated 比例应从 7/20 提升至 10-14/20
- score 应从 36.97 提升至 80-120 区间（不一定超过 iter 2 的 142.32，但应显著改善）

## 8. main_risk
加权和降低质量要求后，策略可能学会"高速撞击但仍获奖"。如果 velocity_quality 权重 0.4 不足以压制高速接触，下一轮需进一步提高速度权重至 0.5-0.6 或引入速度上限硬阈值。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation and soft landing task.
    
    Task: Navigate from random perturbed initial position to target landing pad,
    achieve stable low-speed contact with upright orientation.
    
    Components:
    - goal_approach_progress: dense shaping via distance improvement delta
    - velocity_brake_near_target: quadratic velocity penalty scaled by proximity
    - orientation_stability: quadratic penalty on angle and angular velocity
    - contact_landing_reward: continuous contact reward modulated by quality factors (WEIGHTED SUM)
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
    components["goal_approach_progress"] = 10.0 * distance_delta
    
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
    # Continuous reward for contact, modulated by quality factors
    # Base: any contact (left or right)
    contact_active = max(next_left_contact, next_right_contact)
    
    # Quality modulator 1: velocity (lower is better, bounded to [0,1])
    velocity_quality = max(0.0, 1.0 - velocity_magnitude / 1.0)
    
    # Quality modulator 2: orientation (smaller angle is better, bounded to [0,1])
    angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)
    
    # Quality modulator 3: proximity to target (closer is better, bounded to [0,1])
    proximity_quality = max(0.0, 1.0 - next_distance / 1.0)
    
    # Combined: contact reward scaled by quality factors
    # Using WEIGHTED SUM instead of product to allow partial compensation
    quality_factor = 0.4 * velocity_quality + 0.3 * angle_quality + 0.3 * proximity_quality
    components["contact_landing_reward"] = 5.0 * contact_active * quality_factor
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```
