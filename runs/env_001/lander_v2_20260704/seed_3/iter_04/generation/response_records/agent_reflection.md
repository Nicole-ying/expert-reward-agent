# Response Record

`evidence`：当前轮次 approach_reward 系数 25.0，episode 全部 truncate 在 1000 步，soft_landing_proxy 的 active_rate=0.0（从未触发），实际得分 -10.51 远低于 best 的 149.44；上一轮唯一改动是将 approach_reward 从 1.0 放大到 25.0，直接导致策略从成功着陆退化为全程不终止。

`behavior_diagnosis`：过强的 25x approach 信号使策略在整个 episode 中追逐距离缩减，但到达目标附近后缺乏着陆引导——soft_landing_proxy 因从未双脚触地而归零，agent 在目标附近徘徊或高速掠过，无法完成着陆。

`signal_completeness`：approach_reward 提供过程引导（合理），stability_penalty 提供姿态约束（合理），soft_landing_proxy 承担完成信号职责但 active_rate=0。当前缺的不是组件类型，而是 proxy 因 approach 尺度过强而无法被触发——这是尺度问题而非结构缺失。

`selected_level`：Level 1。soft_landing_proxy 在 best 轮（iter 2）中曾正常激活（sum=1.728），说明其数学形态在合理 approach 尺度下可达；当前 active_rate=0 是 25x approach 的后果，不是 proxy 结构的固有缺陷。

`selected_intervention`：仅修改 approach_reward 系数，从 25.0 降至 3.0。best 的 1.0 已验证可行，25.0 已验证失败，3.0 是介于两者之间的中等强度，保留"尽可能快到达目标"的任务激励同时避免压倒着陆信号。

`falsifiable_hypothesis`：降低 approach_reward 系数后，agent 应有足够空间学习着陆——soft_landing_proxy 的 active_rate 应从 0 回升，episode 应出现 terminated（成功着陆停驻），score 应显著高于 -10.51。

`expected_next_round`：episode_length 应回落至 1000 以下（出现 terminated），soft_landing_proxy active_rate > 0，score 回升至正值区间，approach_reward 的 episode_sum_mean 约为 best 的 3 倍左右（~8-10）。

`main_risk`：3.0 仍可能偏强，若 agent 在目标附近出现轻微振荡但未完全失败，下一轮可能需要进一步降至 1.5-2.0，或对 proxy 做 Level 2 连续化以增强近目标局部反馈。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs indices:
    # 0: x_position, 1: y_position, 2: x_velocity, 3: y_velocity,
    # 4: body_angle, 5: angular_velocity, 6: left_support_contact, 7: right_support_contact
    # goal is at (0,0).

    # 1. Main learning signal: progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = 3.0 * (d_prev - d_next)

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
