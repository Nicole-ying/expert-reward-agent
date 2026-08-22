# 诊断报告

## 1. evidence
外部结果：score=-82.86，19/20 episodes 耗尽 1000 步未完成（truncated），仅 1/20 terminated。组件证据：velocity_brake_near_target 主导 magnitude_share 59.2% 且 active_rate 100%，持续惩罚驱使低速徘徊；goal_approach_progress signed_share 18.1% 远低于 magnitude_share 29.2%，delta 信号正负抵消严重；terminal_soft_landing_bonus active_rate 0.0%，五条件联合从未触发。无上一轮对比（首轮）。

## 2. behavior_diagnosis
策略学会"慢速盘旋拖时间"：保持低速接近目标但从不触地，以回避 velocity_brake_near_target 在接近时激增的惩罚。goal_approach_progress 的 delta 信号零均值无法提供持续方向，terminal_soft_landing_bonus 条件过严零触发，缺乏着陆阶段引导。

## 3. signal_completeness
不完备。缺少可达的着陆过程引导：current 的 terminal_soft_landing_bonus 要求五条件同时满足且 distance<0.5，但 agent 已学会避免接触以回避速度惩罚，该信号永不可达。goal_approach_progress 的 delta 形态在接近目标后振荡，无法区分"接近但未着陆"与"完成着陆"。

## 4. selected_level
Level 2 — sparse_to_dense 变换。触发条件：terminal_soft_landing_bonus active_rate=0.0%，任务关键事件（着陆）几乎不触发，且 agent 行为显示刻意回避接触。需将稀疏的五条件 AND bonus 转换为可达的连续接触引导信号。

## 5. selected_intervention
唯一目标组件：terminal_soft_landing_bonus。具体修改：将五条件稀疏 bonus 改为连续接触奖励 = 基础接触奖励 × 质量调制器（低速、低角度、近目标的连续因子），使 agent 在接近目标时获得稠密正反馈驱动触地探索。系数设为 5.0 以平衡 velocity_brake_near_target 的 -2.0 惩罚，避免接触收益被速度惩罚淹没。

## 6. falsifiable_hypothesis
连续接触奖励提供触地探索的正反馈，应打破"回避接触"的策略平衡。若假设正确，下一轮 active_rate 应显著提升（从 0% 至少到 30%+），terminated episodes 增加（agent 开始尝试触地），episode_length 下降（不再拖满 1000 步），score 初期可能波动但最终应改善。

## 7. expected_next_round
- 新组件（接触奖励）active_rate 从 0.0% 升至 >30%
- terminated episodes 从 1/20 增至 >5/20
- episode_length 从 984.85 降至 <800
- score 可能在 [-150, 0] 范围波动（探索代价），但 best episode 应出现 >-50 的轨迹
- velocity_brake_near_target magnitude_share 从 59.2% 降至 <40%（接触信号分流占比）

## 8. main_risk
接触奖励可能奖励"任意接触"而非"目标垫接触"，若 agent 学会撞地板其他区域刷分。缓解：质量调制器中的 near_target 因子（distance<1.0 时线性衰减）限制奖励范围，只在目标附近触地才有显著收益。

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