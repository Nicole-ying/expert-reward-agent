# Response Record

# 证据分析

**evidence**: iter 2 达到 142.3 分，终止率未知但 len=721 显著短于基线；iter 3~6 连续四轮在 best 骨架上调整 contact_landing_reward 的质量门控形式（乘积 vs 加性平均），score 全部劣于 best，且 iter 5/6 中 goal_approach_progress 的 per-step 均值（0.230/951.8=0.00024 和 0.030/784.4=0.00004）远低于 best 的 0.027/721.1=0.000037，说明调整质量门控后 agent 学习到了更低效的接近行为。contact_landing_reward 在所有迭代中 per-step 均值约 2.1~2.3/step，episode_sum 509，占比 90.1%，但 active_rate 仅 15.9%，说明 agent 学会了在少数步内密集触发接触奖励，而不是高效到达目标。

**behavior_diagnosis**: 策略已学会在接触垫附近反复触发 contact_landing_reward（active_rate 15.9%但占总奖励 90%），而不是快速接近目标后稳定着陆。goal_approach_progress 的 per-step 均值从 best 的 0.000037 降至 current 的 0.000038（几乎持平但绝对值极低），说明接近效率未改善。len=784 高于 best 的 721，terminated=11/20 说明仅 55% 回合正常终止，剩余 45% 超时截断，表明策略在接触后未能快速收敛到稳定状态。

**signal_completeness**: 当前骨架包含接近引导（goal_approach_progress）、接触奖励（contact_landing_reward）、速度/姿态约束（velocity_brake_near_target、orientation_stability），职责完备。但 contact_landing_reward 的激活模式（15.9% active_rate 贡献 90% 奖励）表明其数学形态鼓励了"反复触碰"而非"稳定保持接触"，与任务要求的"软着陆后静止"脱节。

**selected_level**: Level 2 — 数学形态修正。contact_landing_reward 的质量门控（无论乘积或加性）均未解决根本问题：它奖励"接触发生"这一瞬时事件，而非"保持稳定接触状态"。当 agent 学会在垫附近反复弹跳触发接触时，该组件会持续给予高奖励，即使 agent 从未真正稳定着陆。需要将 contact_landing_reward 从"事件触发"改为"状态持续"形式。

**selected_intervention**: 修改 contact_landing_reward 的数学形态，从 `contact_active * quality_factor` 改为双腿同时接触时的持续状态奖励，并要求质量条件联合满足后才给予基础奖励，避免单腿弹跳或不稳定接触时获奖。具体方案：仅在 `left_contact > 0 且 right_contact > 0` 时给予基础分（确保双腿着地），并将质量门控从"调制系数"改为"激活条件"——只有当速度、姿态、位置同时达标时才给予奖励，否则为 0。这将迫使 agent 学习"稳定保持双腿接触且满足着陆条件"，而非"反复单腿触碰"。

**falsifiable_hypothesis**: 将 contact_landing_reward 改为要求双腿同时接触且质量条件联合满足后才激活，应使 agent 放弃"单腿弹跳刷分"策略，转而学习"双腿稳定着陆后保持静止"。预期 contact_landing_reward 的 active_rate 会从 15.9% 下降（因条件更严格），但 episode_sum 占比应保持高位（因每次激活时持续时间更长）。同时 goal_approach_progress 的 per-step 均值应上升（因 agent 需要更快到达目标以进入稳定接触阶段）。

**expected_next_round**: (1) contact_landing_reward 的 active_rate 下降至 5%~10%（因要求双腿同时接触且质量达标），但 episode_sum 占比维持 70%~85%（因激活后持续给予奖励）；(2) goal_approach_progress 的 per-step 均值上升至 0.00005~0.0001（agent 需更快接近目标）；(3) terminated 比例上升至 14/20~16/20（agent 学会稳定着陆后更快触发 body_not_awake 终止）；(4) score 上升至 160~180（接近或超越 best）。

**main_risk**: 质量条件过严可能导致 contact_landing_reward 完全不激活（active_rate → 0），使 agent 失去接触引导信号，退化为纯接近行为。若下一轮 active_rate < 3% 且 score 下降，需放松质量门控（例如改为阈值从严格的 0.3 放宽至 0.5）。

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
    - contact_landing_reward: continuous reward for STABLE dual-leg contact under quality gate
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
    
    dual_contact = (next_left_contact > 0.0) and (next_right_contact > 0.0)
    
    velocity_ok = velocity_magnitude < 0.4
    angle_ok = abs(next_angle) < 0.3
    proximity_ok = next_distance < 0.6
    
    if dual_contact and velocity_ok and angle_ok and proximity_ok:
        components["contact_landing_reward"] = 8.0
    else:
        components["contact_landing_reward"] = 0.0
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```
