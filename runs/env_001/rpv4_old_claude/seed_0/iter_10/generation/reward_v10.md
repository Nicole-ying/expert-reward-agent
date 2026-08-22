# 奖励函数诊断与修订

## 1. evidence
Subagent 明确指出 iter 8→9 的统计口径差异掩盖了真实行为变化：contact 从"稠密低质"（49.5% active）转向"稀疏高质"（6.9% active），episode_sum 从 2.44×848 涨至 140.04，但 score 暴跌 10.4 分。这证明策略找到捷径：用高速撞地触发 contact_active=1.0，但 quality_factor 三因子相乘后趋零。历史最佳 iter 2（score=142.32）同样使用乘积形式但系数为 5.0，当前降至 4.0 后策略崩溃，说明乘积形式本身在边界条件下脆弱。

## 2. behavior_diagnosis
策略收敛到病态局部解：快速接近目标后高速撞地，满足 contact_active 但牺牲 velocity/angle/proximity quality，导致 contact reward 名义值高（140.04）但实际引导失效，12/20 episode terminated 中部分是撞毁（score range [-132.8, 181.4] 极端波动证实）。

## 3. signal_completeness
当前骨架缺失**独立的安全着陆判定**。contact_landing_reward 试图用乘积形式同时表达"是否接触"和"接触质量"，但允许策略先满足前者（简单）再忽略后者（困难）。任务需要"低速+正姿+接触"的联合满足信号，而非可被解耦利用的连续调制器。

## 4. selected_level
**Level 2：数学形态修正**

触发条件：contact_landing_reward 的乘积形式在极端情况下允许 quality_factor→0 但 contact_active=1.0，导致策略学会"撞地捷径"。这不是系数问题（iter 2 的 5.0 和 iter 9 的 4.0 差异不足以解释 143 分的崩溃），而是数学形态在边界条件下传达了错误激励。

## 5. selected_intervention
**目标组件**：contact_landing_reward

**具体修改**：从"contact_active × quality_factor"的连续调制器改为**稀疏事件 bonus**，要求同时满足：
1. 接触（left_contact OR right_contact = 1.0）
2. 速度安全（velocity_magnitude < 0.5）
3. 姿态稳定（|angle| < 0.3）
4. 目标附近（distance < 0.8）

使用阶跃函数而非连续乘积，防止策略解耦利用。保留原骨架其他三个组件不变（goal_approach_progress 提供密集引导，velocity_brake 和 orientation_stability 提供约束）。

## 6. falsifiable_hypothesis
稀疏联合条件将迫使策略在接触前同时优化速度、姿态和位置，而非先撞地再事后补救。预期 contact 的 active_rate 从 6.9% 降至 1-3%（仅在真正满足条件时触发），但单次触发的价值更高且对齐任务目标。如果假设错误，下一轮将看到 active_rate 接近 0%（条件过严无法触发）或 score 进一步下跌（稀疏信号不足以引导学习）。

## 7. expected_next_round
- contact_landing_reward 的 active_rate 应降至 1-3%，episode_sum_mean 降至 20-50（单次 bonus 高但触发少）
- score 应回升至 50-100（部分恢复 iter 2-4 水平），如未回升则证明稀疏信号过弱
- terminated 比例应维持或略增（12/20→14/20），且 score_range 波动收窄（说明撞毁减少）
- goal_approach_progress 应保持稳定（per-step 0.01-0.03），作为主要密集引导

## 8. main_risk
稀疏条件可能过严，导致训练早期完全无法触发 bonus，策略退化为仅依赖 goal_approach_progress 的接近行为而不尝试着陆。如果下轮 contact active_rate < 0.5% 且 score 无改善，需放宽阈值（velocity < 0.8, angle < 0.5, distance < 1.2）或引入连续 pre-bonus 作为过渡。

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
    - contact_landing_reward: sparse bonus for safe landing (joint satisfaction of contact + velocity + angle + proximity)
    """
    
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle, next_ang_vel = next_obs[4], next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    components = {}
    
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (next_x_pos**2 + next_y_pos**2) ** 0.5
    distance_delta = current_distance - next_distance
    components["goal_approach_progress"] = 10.0 * distance_delta
    
    velocity_magnitude = (next_x_vel**2 + next_y_vel**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + next_distance)
    components["velocity_brake_near_target"] = -2.0 * proximity_factor * velocity_magnitude**2
    
    angle_penalty = next_angle**2
    ang_vel_penalty = next_ang_vel**2
    components["orientation_stability"] = -1.0 * (angle_penalty + 0.5 * ang_vel_penalty)
    
    contact_active = max(next_left_contact, next_right_contact)
    velocity_safe = 1.0 if velocity_magnitude < 0.5 else 0.0
    angle_stable = 1.0 if abs(next_angle) < 0.3 else 0.0
    proximity_close = 1.0 if next_distance < 0.8 else 0.0
    
    safe_landing_achieved = contact_active * velocity_safe * angle_stable * proximity_close
    components["contact_landing_reward"] = 50.0 * safe_landing_achieved
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```