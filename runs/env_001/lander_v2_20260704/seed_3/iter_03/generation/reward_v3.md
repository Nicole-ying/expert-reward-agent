`evidence`：20/20 terminated（无truncated），score 从 -95 跃升至 149，但 episode_length=565 偏长。soft_landing_proxy 占 97.4% magnitude share（active_rate 10.9%），而 approach_reward 虽 active_rate 98.8% 但仅贡献 1.8% magnitude，每步均值约 0.002，近乎被淹没。

`behavior_diagnosis`：agent 已学会最终着陆并稳定停驻，但移动极其缓慢——approach 信号太弱，几乎所有学习压力来自仅在平台附近激活的 soft_landing_proxy，缺乏“尽快到达”的驱动。

`signal_completeness`：approach_reward 数学形态正确（连续 delta-distance、符号正向），但尺度严重不足；stability_penalty 存在但可忽略；soft_landing_proxy 提供完成信号但仅末端有效。缺少引擎推力惩罚，但当前优先修复主导信号尺度。

`selected_level`：Level 1。approach_reward 的职责、符号和数学形态均合理，证据仅指向其相对 soft_landing_proxy 过弱（累计比 ≈1:67），不需要结构变换。

`selected_intervention`：将 approach_reward 乘以系数 25.0，其他组件不变。预期总 approach_reward 从 ~1.07 升至 ~26.75，约为 soft_landing_proxy 的 37%，使进度信号在早期学习中具备实际引导力。

`falsifiable_hypothesis`：更强的 progress 梯度应促使 agent 更快靠近平台，从而降低 episode_length；若 hypothesis 正确，episode_length 将下降，approach_reward 的 magnitude_share 升至 15-35%，且 terminated 比例应保持或改善。

`expected_next_round`：episode_length 下降至 300-450，approach_reward magnitude_share 升至 15-35%，soft_landing_proxy share 降至 55-75%，terminated 保持 100%，score 持平或改善。

`main_risk`：25x 乘数可能使 agent 过快冲向平台导致撞击失败（early terminal 增加），或牺牲着陆姿态质量。若出现此情况，下一轮应回退系数并考虑 Level 2 全局门控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs indices:
    # 0: x_position, 1: y_position, 2: x_velocity, 3: y_velocity,
    # 4: body_angle, 5: angular_velocity, 6: left_support_contact, 7: right_support_contact
    # goal is at (0,0).

    # 1. Main learning signal: progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = 25.0 * (d_prev - d_next)

    # 2. Stability constraint: light penalty on large speeds, tilt, and angular velocity.
    speed = abs(next_obs[2]) + abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    w_speed = 0.001
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = -w_speed * speed - w_angle * angle - w_angvel * ang_vel

    # 3. Soft landing proxy: boost when almost on target, slow, upright, and both feet contact.
    dist = d_next
    contact_both = next_obs[6] * next_obs[7]
    k_dist = 2.0
    k_speed = 1.0
    k_angle = 5.0
    w_proxy = 5.0
    soft_landing_proxy = (w_proxy * contact_both *
                          (2.718281828 ** (-k_dist * dist)) *
                          (2.718281828 ** (-k_speed * speed)) *
                          (2.718281828 ** (-k_angle * angle)))

    total_reward = approach_reward + stability_penalty + soft_landing_proxy

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```