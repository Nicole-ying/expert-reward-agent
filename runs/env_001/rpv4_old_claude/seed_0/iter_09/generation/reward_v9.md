# 诊断与干预决策

## 1. evidence
Subagent 明确指出 contact_landing_reward 的 per-step 量级（131.1 when active）比 goal_approach（0.011）大 11,918 倍，导致"撞击式接触"而非"稳定停留"。历史最佳 iter2（142.32 分，len=721）使用系数 5.0，当前 iter8（8.81 分，len=848）使用系数 6.0，提升系数后 contact 激活率从 49.5%（iter7 推测）降至 3.4%，terminated 从 0→9，但 score 崩溃。iter7→iter8 代码完全相同却结果巨变，说明训练随机性大或策略进入局部最优。

## 2. behavior_diagnosis
策略学会短暂高质量接触（terminated=9/20）但立即终止，未实现持续停留。contact_landing_reward 占总奖励 69.3% 但仅在 3.4% 步数激活，quality_factor 的三项乘积在接近目标时过度压缩，使策略放弃维持接触。

## 3. signal_completeness
缺少"保持接触状态"的持续激励。当前 contact_landing_reward 奖励"进入接触瞬间"而非"停留在接触"，且 quality_factor 对持续接触的敏感度不足。

## 4. selected_level
Level 1：系数调整。

触发条件：历史最佳 iter2 证明当前骨架核心成立（142.32 分 > target 70%），但 iter8 的系数 6.0 过强导致量级失衡。Subagent 建议降低 contact base reward 至与 goal_approach 同量级（0.1-1.0），但考虑到 iter2 使用 5.0 已接近成功，我选择更保守的回退：将系数从 6.0 降至 4.0，使 per-step 量级从 131.1 降至约 87.4（假设 active_rate 保持），仍比 goal_approach 的 0.011 强但不至于完全淹没过程信号。

## 5. selected_intervention
唯一目标组件：contact_landing_reward

具体修改：将系数从 6.0 降至 4.0。不改变 quality_factor 数学形态（保持三项几何平均），不调整其他组件。

## 6. falsifiable_hypothesis
降低 contact 奖励峰值将减少"撞击后立即终止"的诱惑，使策略有动力通过 goal_approach（每步累积）和 velocity_brake（持续约束）优化接近轨迹。预期 contact 激活率上升（agent 愿意维持接触更长时间），terminated 数量保持或略降，score 向 iter2（142.32）回升。

## 7. expected_next_round
- score：从 8.81 回升至 80-140 区间
- len：从 848 降至 700-800（更快完成任务）
- terminated：保持 9/20 或略降至 6-8/20（稳定停留代替快速终止）
- contact_landing_reward active_rate：从 3.4% 上升至 8-15%（持续接触时长增加）
- contact_landing_reward episode_sum_mean：从 111.1 降至 60-90（单次激活回报降低但总时长增加）
- goal_approach_progress episode_sum_mean：保持或略升（不再被完全淹没）

## 8. main_risk
系数 4.0 可能仍过强。如果 per-step 量级依然压倒 goal_approach，策略可能继续"撞击式接触"，只是峰值略低。备选方案是进一步降至 2.0-3.0，但当前先验证 4.0 是否足够（iter2 的 5.0 曾成功，4.0 是向下微调）。

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
    
    velocity_quality = max(0.0, 1.0 - velocity_magnitude / 1.0)
    angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)
    proximity_quality = max(0.0, 1.0 - next_distance / 1.0)
    
    quality_factor = (velocity_quality * angle_quality * proximity_quality) ** 0.333
    components["contact_landing_reward"] = 4.0 * contact_active * quality_factor
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```