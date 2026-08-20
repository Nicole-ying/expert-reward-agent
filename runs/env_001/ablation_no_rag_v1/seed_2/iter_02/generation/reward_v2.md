```text
evidence: 
上一轮全 episode 在 1000 步截断，无成功着陆；progress_reward episode_sum_mean=1.012 说明 agent 在持续接近目标，但 stable_landing_reward 的 active_rate=0、均值 0，因为乘积中 contact_factor 始终为 0 导致整个组件无梯度输入。

behavior_diagnosis: 
agent 学会了向目标中心移动，但无法完成最终的着陆接触，在接近中心后可能由于缺乏减速、摆正和接触激励而长期徘徊，最终被时间截断。

signal_completeness: 
缺少引导着陆接触和成功完成的信号；原来的 stable_landing_reward 设计成接触因子与指数乘积，结果因接触始终为 0 而完全塌缩，丧失了到达目标后精细控制的学习信号。

selected_level: 
Level 2，因为证据直接否定了当前组件的数学形态（乘积塌缩），且必要着陆引导职责缺失。

selected_intervention: 
将 stable_landing_reward 从乘积形式改为加性组合，具体包括：
- 距离门控速度惩罚（-0.1*speed 当 dist<2）
- 距离门控角度惩罚（-0.1*abs(angle) 当 dist<2）
- 接触变化奖励（每脚从未接触到接触得 +10）
三部分之和作为新的 stable_landing_reward。

falsifiable_hypothesis: 
新增的速度/角度惩罚提供精细控制的稠密信号，接触变化奖励提供稀疏但关键的着陆激励，应促使 agent 在接近中心后减速、竖直，并主动尝试使支撑脚接触，从而在有限步内完成着陆。

expected_next_round: 
stable_landing_reward active_rate > 0，出现接触变化奖励的尖峰，episode 长度显著缩短（部分成功着陆），score 大幅提升。

main_risk: 
速度/角度惩罚虽然通过近场门控避免早期干扰，但若门控阈值 2.0 不合适可能导致精细控制过早或过晚；单脚接触的一次性奖励可能让 agent 止于单脚接触而不追求双脚稳定，但通常成功着陆需要双脚接触，环境会终止，因此风险较低。
```

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：基于位置进步的 potential-based shaping
    dist_old = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_new = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_old - dist_new

    # 着陆引导与接触奖励
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact_old = obs[6]
    right_contact_old = obs[7]
    left_contact_new = next_obs[6]
    right_contact_new = next_obs[7]

    # 进入着陆区域（靠近中心）才施加精细控制
    dist = (x ** 2 + y ** 2) ** 0.5
    near_pad = float(dist < 2.0)

    # 速度惩罚：鼓励静止
    speed = (vx ** 2 + vy ** 2) ** 0.5
    velocity_penalty = -0.1 * speed * near_pad

    # 角度惩罚：鼓励竖直
    angle_abs = abs(angle)
    angle_penalty = -0.1 * angle_abs * near_pad

    # 接触变化奖励（一次性，鼓励支撑脚接触）
    contact_change = max(0.0, left_contact_new - left_contact_old) + max(0.0, right_contact_new - right_contact_old)
    contact_bonus = 10.0 * contact_change

    stable_landing_reward = velocity_penalty + angle_penalty + contact_bonus

    total_reward = progress_reward + stable_landing_reward

    components = {
        "progress_reward": progress_reward,
        "stable_landing_reward": stable_landing_reward
    }

    return float(total_reward), components
```