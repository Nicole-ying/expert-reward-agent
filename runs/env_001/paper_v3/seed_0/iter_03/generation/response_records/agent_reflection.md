# Response Record

1. `evidence`: 最终策略评估中 landing_proxy 均值 946.38 远大于其他组件，占 signed_share 82.5%；得分 183.64 距离目标 200 还差 16.36；episode 平均长度 436.45，17/20 以 terminated 结束，表明大部分成功着陆但策略可能徘徊缓慢；缺失明确的 safe_landing_confirmation 稀疏奖励。
2. `behavior_diagnosis`: 策略学会了着陆，但 landing_proxy 作为持续稠密奖励鼓励在目标区域保持良好状态，导致策略可能满足于接近目标徘徊而非尽快完成，外部得分未达目标。
3. `signal_completeness`: 具备位置靠近、减速、姿态稳定等引导信号，但缺少明确的“成功完成”稀疏奖励，使得完成信号被淹没在持续的 landing_proxy 中。
4. `selected_level`: Level 2 — 必要职责 safe_landing_confirmation 缺失，需以稀疏事件形式补充；非纯尺度问题，因即便调低 landing_proxy 也无法提供“完成”的明确激励。
5. `selected_intervention`: 新增 success_bonus 组件，当同时满足接近目标（dist < 0.1）、极低速度（speed < 0.1）、姿态水平（|angle| < 0.1）且双支撑接触时给予固定大额奖励 +100。
6. `falsifiable_hypothesis`: 明确的完成奖励会引导策略尽快触发 settled 条件，缩短 episode 长度并减少徘徊，从而提升外部任务得分（可能因更快完成或更高效而获益）。
7. `expected_next_round`: 下一轮应观察到 episode 平均长度下降，terminated 比例保持或提高，外部 score 提升，同时 success_bonus 的 active_rate 极低（仅最后一步触发）但 episode_sum_mean 体现明显。
8. `main_risk`: 若 settled 阈值设置不当或 +100 激励过大，可能导致策略过早试图“宣告成功”而冲撞地面，但 landing_proxy 中的速度/角度惩罚可抑制此行为；不会引入崩溃式失败。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: discourages high speed when close to the target
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: sum of bounded factors rewarding simultaneous proximity,
      low speed, upright posture, and ground contact
    - success_bonus: large sparse reward for achieving settled conditions on target
    """
    # next_obs unpacking
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Euclidean distance to target
    dist = (x**2 + y**2) ** 0.5

    # Component A: distance penalty (core progress signal)
    w_dist = 1.0
    distance_penalty = -w_dist * dist

    # Component B: velocity penalty (damped by distance to target)
    w_vel = 0.2
    gate = 1.0 / (1.0 + dist)          # near target -> gate approx 1; far away -> gate approx 0
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # Component C: orientation stabilization penalty
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # Component D: soft landing proxy (sum of bounded factors, non-collapsing joint)
    w_landing = 3.0
    # proximity factor: 1 at dist=0, 0 when dist >= 2.5
    prox_reward = max(0.0, 1.0 - dist / 2.5)
    # speed factor: 1 at speed=0, 0 when speed >= 2.0 m/s
    speed = (vx**2 + vy**2) ** 0.5
    vel_reward = max(0.0, 1.0 - speed / 2.0)
    # angle factor: 1 at angle=0, 0 when |angle| >= 0.5 rad (~28.6 deg)
    angle_reward = max(0.0, 1.0 - abs(angle) / 0.5)
    # contact factor: 0, 0.5, or 1.0 depending on single/double contact
    contact_reward = (left_contact + right_contact) / 2.0
    # sum of four independent bounded factors, averaged
    landing_proxy = w_landing * (prox_reward + vel_reward + angle_reward + contact_reward) / 4.0

    # Component E: success bonus (sparse, triggered only when fully settled on target)
    w_success = 100.0
    settled_dist = dist < 0.1
    settled_speed = speed < 0.1
    settled_angle = abs(angle) < 0.1
    settled_contact = (left_contact + right_contact) == 2.0
    success_bonus = w_success if (settled_dist and settled_speed and settled_angle and settled_contact) else 0.0

    # Total reward
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy + success_bonus

    # Component dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy,
        "success_bonus": success_bonus
    }

    return float(total_reward), components
```
