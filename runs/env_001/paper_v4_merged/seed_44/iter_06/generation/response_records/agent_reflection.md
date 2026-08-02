# Response Record

1. `evidence`：上一轮得分222.88未刷新best 250.95，同骨架已连续3轮（iter3-5）迭代；contact_reward占据47.1% signed_share但active_rate仅44.1%且training中低至21.8%，而progress均值从0.099降至0.078、len从333.85升至552.5，强烈暗示策略通过缓慢漂浮获取持续接触奖励导致proxy exploitation。
2. `behavior_diagnosis`：策略学会了在中心附近极慢悬停以持续触发contact_reward，从而延长episode、降低per-step进度但保持高分，但牺牲了更快着陆的效率与稳定性。
3. `signal_completeness`：所有obs维度均已使用，引导接近、减速、姿态、腿接触的职责均有覆盖，但接触奖励的持续发放与任务成功（一次性安全着陆）失配，形成可被利用的错误信号。
4. `selected_level`：Level 2结构变换——contact_reward属于“占据好状态即持续获奖”的持续奖励形态，触发proxy徘徊；将其改为一次性着陆改善量（state→improvement）。
5. `selected_intervention`：移除原contact_reward，替换为基于双腿接触上升沿的一次性着陆奖励，仅在从非双脚接触到双脚接触的瞬间给予，并用接近度、姿态质量、速度质量乘积决定奖励值。
6. `falsifiable_hypothesis`：将接触奖励改为一次性事件会消除持续悬停的动机，迫使策略更快完成软着陆，从而缩短episode length并提升或恢复最高得分。
7. `expected_next_round`：episode length应下降至300-400区间，progress的episode_sum_mean回升至0.09以上，contact_landing_bonus的active_rate极低但episode_sum_mean为正值，总得分应至少恢复到250以上。
8. `main_risk`：一次性奖励过于稀疏可能导致着陆行为退化、早期探索不足从而得分暂时下降；如果proximity gate设置过严（dist/0.5限制）可能导致奖励从未触发，需后续观察并调整阈值。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Progress toward center
    x_curr, y_curr = obs[0], obs[1]
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    x_next, y_next = next_obs[0], next_obs[1]
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = dist_curr - dist_next

    # Attitude and angular velocity penalty
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    angle_penalty = body_angle ** 2 + angular_vel ** 2

    # Soft landing guide
    dist_factor = 2.718281828 ** (-dist_next / 0.5)
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    speed_factor = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 1.0)
    landing_reward = dist_factor * speed_factor

    # One-time landing bonus: reward the moment both legs transition from non-contact to contact
    prev_contact = obs[6] * obs[7]          # 0 or 1
    next_contact = next_obs[6] * next_obs[7]
    contact_rising = max(0.0, next_contact - prev_contact)  # 1 on rising edge only

    proximity_factor = max(0.0, 1.0 - dist_next / 0.5)
    attitude_quality = max(0.0, 1.0 - (abs(body_angle) + abs(angular_vel)) / 0.5)
    speed_quality = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.5)
    contact_landing_bonus = 20.0 * contact_rising * proximity_factor * attitude_quality * speed_quality

    total = (
        10.0 * progress
        - 0.5 * angle_penalty
        + 0.01 * landing_reward
        + contact_landing_bonus
    )

    components = {
        "progress": 10.0 * progress,
        "angle_penalty": -0.5 * angle_penalty,
        "landing_reward": 0.01 * landing_reward,
        "contact_landing_bonus": contact_landing_bonus
    }

    return float(total), components
```
