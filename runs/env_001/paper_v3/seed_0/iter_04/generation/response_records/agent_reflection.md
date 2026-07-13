# Response Record

1. `evidence`：最终策略得分-57.36，terminated 15/20，episode长度628步；landing_proxy每回合平均积累1236（magnitude_share 78%），success_bonus active_rate=0%从未触发；距离惩罚占比21%；agent在628步里未能稳定着陆，外部得分极差。

2. `behavior_diagnosis`：agent在目标附近中等距离徘徊，大量积累landing_proxy的稠密奖励（四个独立因子只要部分满足就能得分），但始终未达到成功条件的严格阈值，导致proxy farming而任务失败。

3. `signal_completeness`：距离引导、速度抑制、姿态稳定和双接触信号均可用，但landing_proxy的独立加和结构允许通过部分满足条件获得高额奖励，造成信号与完成条件的错位。

4. `selected_level`：Level 2，触发条件为“proxy提高但外部任务不升”，对应proxy_to_completion_alignment变换；当前landing_proxy在部分满足条件时持续给予奖励，引导出不利于完成的徘徊行为。

5. `selected_intervention`：唯一干预目标是landing_proxy组件；将其从无条件稠密求和改为门控触发：仅当proximity、speed、angle三个因子均超过0.6阈值时才计算landing_proxy，否则返回0，从而迫使agent必须同时满足接近、低速、姿态良好的前提才能获得该奖励。

6. `falsifiable_hypothesis`：取消低质量状态下的landing_proxy后，agent无法再通过徘徊积累大量回报，必须驱动距离惩罚、速度惩罚和姿态惩罚去真正接近并稳定着陆，从而增加触发success_bonus的概率，提升外部得分。

7. `expected_next_round`：landing_proxy的episode_sum_mean应显著下降（不再凭空积累），episode长度可能先下降后回升，success_bonus active_rate从0上升，外部score提高。

8. `main_risk`：门控阈值可能使landing_proxy早期过于稀疏，导致学习速度变慢或陷入局部最优；如果距离惩罚本身引导不足，agent可能无法学会同时满足所有前提条件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: discourages high speed when close to the target
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: gated joint reward that only activates when proximity, speed, and angle
      are all sufficiently good, encouraging simultaneous requirements
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

    # Component D: gated soft landing proxy (only active when basic conditions are met)
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

    # gate: landing_proxy is zero unless proximity, speed, and angle all pass a threshold
    gate_open = (prox_reward > 0.6) and (vel_reward > 0.6) and (angle_reward > 0.6)
    if gate_open:
        landing_proxy = w_landing * (prox_reward + vel_reward + angle_reward + contact_reward) / 4.0
    else:
        landing_proxy = 0.0

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
