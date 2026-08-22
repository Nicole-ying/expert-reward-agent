# 奖励函数诊断与修订

## 1. evidence
Subagent 明确指出 iter6→7 的关键退化：contact_landing_reward 从连续质量调制（iter2 最佳：5.0 * contact_active * quality_factor^0.333）变为严格全有全无门槛（需同时满足 dual_contact + velocity<0.4 + angle<0.3 + distance<0.6），导致激活率从 50.3%（iter2）→ 39.3%（iter6）→ 0.0%（iter7）。velocity_brake 的 proximity_factor 设计在接近目标时放大速度惩罚（-0.023/step，占 58.1% magnitude），与"慢速接近着陆"相悖。全部 20 个 episode 跑满 1000 步未着陆（terminated=0/20），策略学会悬停而非着陆。

## 2. behavior_diagnosis
策略陷入"悬停陷阱"：contact_landing_reward 的严格门槛让任何不完美的着陆尝试得不到反馈，velocity_brake 的 proximity_factor 惩罚接近目标的速度，两者共同教会策略"保持距离+空中稳定"而非"慢速接近+着陆"。

## 3. signal_completeness
- **着陆引导职责**：已具备但失效（contact_landing_reward 门槛过严）
- **接近引导职责**：已具备且健康（goal_approach_progress）
- **速度约束职责**：已具备但错位（velocity_brake 惩罚接近而非高速）
- **姿态约束职责**：已具备且健康（orientation_stability）

## 4. selected_level
**Level 2（数学形态修正）**。Subagent 诊断显示 contact_landing_reward 的门槛化是病因——iter2 最佳版本的连续调制形态是正确的，iter7 的门槛形态教错了行为。这不是量级问题（contact 的 per-step 量级在 iter2 是 2.104/721.1=0.00292，完全健康），而是数学形态从连续退化为稀疏导致的激活崩溃。

## 5. selected_intervention
**唯一目标组件：contact_landing_reward**

**具体修改**：从 iter7 的严格门槛（dual_contact AND velocity<0.4 AND angle<0.3 AND distance<0.6 → 8.0 else 0.0）回退到 iter2 最佳的连续质量调制形态（5.0 * contact_active * quality_factor^0.333），但提升系数从 5.0→6.0 以补偿 velocity_brake 的压制效应。保持 iter2 的三个质量调制器（velocity_quality、angle_quality、proximity_quality）及其 bounded 形态，保持几何平均（^0.333）以避免乘积塌缩。

**不动其他组件**：goal_approach_progress、velocity_brake_near_target、orientation_stability 保持 iter7 参数不变（尽管 velocity_brake 的 proximity_factor 设计存在结构性问题，但 Subagent 将其列为次要干预目标，且 iter2 同样使用该设计但取得最佳分数，说明在 contact 健康时 velocity_brake 可被容忍）。

## 6. falsifiable_hypothesis
恢复连续 contact 引导后，策略将重新获得着陆尝试的密集反馈。contact_landing_reward 的激活率应从 0.0% 恢复至 ≥30%，terminated 应从 0/20 恢复至 ≥8/20，episode_length 应从 1000 降至 <850。提升系数 6.0 应让 contact 的 per-step 贡献（预期 ~0.004）超过 velocity_brake 的压制（-0.023），推动策略从悬停转向接触尝试。

## 7. expected_next_round
- **contact_landing_reward active_rate**: 0.0% → ≥30%（恢复连续激活）
- **terminated**: 0/20 → ≥8/20（重新出现着陆成功）
- **episode_length**: 1000 → <850（减少悬停时间）
- **score**: -31.83 → ≥80（接近 iter2 的 142.32，但系数提升可能略优）

## 8. main_risk
如果 velocity_brake 的 proximity_factor 压制过强，提升 contact 系数至 6.0 可能不足以翻转激励，策略可能在"悬停"与"硬着陆"之间振荡而非软着陆。若此风险实现，下一轮需 Level 2 修正 velocity_brake（移除 proximity_factor 或改为鼓励慢速接近的形态）。

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
    components["contact_landing_reward"] = 6.0 * contact_active * quality_factor
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```