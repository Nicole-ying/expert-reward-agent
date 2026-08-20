`evidence`：上一轮 score=-110.84，20/20 均为 early terminal（平均步长 68.4），landing_proxy active_rate 仅 1.2% 且 episode_sum_mean=0.74，distance_penalty 占 magnitude_share 84%；说明 agent 在距离引导下接近目标但无法减速着陆，最终高速撞击或异常 settled 触发终止。
`behavior_diagnosis`：策略被 distance_penalty 推向目标区域，但 landing_proxy 因四因子连乘几乎永远为零，无法为减速、调姿、双足触地提供任何正向学习梯度，导致 agent 以不可控姿态和速度冲向目标并 crash。
`signal_completeness`：position_approaching 和 orientation_stabilization 职责可达且符号正确；safe_landing_confirmation 虽在代码中存在，但因 product collapse 实际不可达，缺少有效的着陆正向引导。
`selected_level`：Level 2 — product_to_noncollapsing_joint。landing_proxy 四因子指数连乘导致 active_rate≈1.2%，符合 product collapse 诊断，需将乘积形态改为非塌缩联合形式。
`selected_intervention`：仅修改 landing_proxy 组件。将 `prox_factor * vel_factor * angle_factor * double_contact` 乘积替换为四个独立 bounded 因子的等权求和（proximity, speed, angle, contact 各用线性饱和 max(0,1-x/D)），消去指数连乘导致的塌缩。
`falsifiable_hypothesis`：求和形态使 landing_proxy 在部分条件满足时即可输出非零梯度（例如靠近但速度仍高、或姿态好但未接触），active_rate 应显著上升，landing_proxy 的 episode_sum_mean 和 magnitude_share 提高，agent 开始获得着陆相关正向反馈从而减速靠近目标而非撞击。
`expected_next_round`：terminated 率可能仍高但 episode_length 应有所增加（不再立即撞击），landing_proxy active_rate 从 1.2% 升至 >30%，其 magnitude_share 从 0.9% 升至 5%-15%，score 应开始向正向移动。
`main_risk`：求和形态允许各因子互相补偿，若 agent 仅在远距离保持低速小角度即获取中等 landing_proxy，可能产生徘徊行为；proximity 因子在线性归零边界（dist≈2.5）附近的梯度断续也可能造成边界策略不稳定。

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

    # Total reward
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy

    # Component dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components
```