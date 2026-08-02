# Response Record

1. `evidence`：terminated 仅 1/20（5%），绝大多数 episode 超时存活，说明 agent 在目标平台附近反复接触但未触发稳定着陆终止；contact_reward 贡献 98.7% 份额，progress_delta 虽全活跃但份额仅 1%，agent 缺乏完成稳定状态的激励；orientation_penalty 与 speed_penalty 几乎未触发（active_rate 1.9%/3.3%），但对行为几乎无影响，属于僵尸组件；上轮添加 contact_reward 使分数从 128.5 升至 144.8，效果正向但未解决稳定着陆问题。
2. `behavior_diagnosis`：Agent 已学会频繁触地（双脚接触概率高），在目标区域附近小幅度摆动，既不摔毁也不出界，但未能将速度、角速度、姿态归零以触发「body_not_awake_or_settled」终止，导致长期徘徊至超时，累计分数被 contact 奖金主导但缺乏完成终态的高价值奖励。
3. `signal_completeness`：当前信号缺失「接近稳定着陆并完成」的奖励——已具备距离、速度、角度、接触等观测，但缺少将这些条件同时满足时的正向复合激励，导致 agent 无动力将状态维持在低速度、低角度、中心、双脚接触的状态，也就无法触发环境内建的成功终止。
4. `selected_level`：Level 2（结构变换），因为信号缺口明确（缺少完成状态激励），需新增一个基于观测组合的持续奖励组件，不能仅靠尺度修复现有组件。
5. `selected_intervention`：新增组件 `landing_progress`，基于距离接近中心、线速度小、角速度小、角度小、双脚接触的乘积因子，给予每步 0.15 的奖励，驱动 agent 持续保持稳定着陆条件并最终触发自然终止。
6. `falsifiable_hypothesis`：添加 `landing_progress` 后，agent 将学会在接近目标后迅速减速、摆正姿态、保持双脚着地，从而使 episode 终止提前（len 下降，terminated 比例上升），同时由于完成状态的奖励积累，score 将显著提升并向 200 靠拢。
7. `expected_next_round`：下轮 terminated 占比应从 5% 升至 >20%，episode_length 中位数下降至 600–800，score 提升至 170–190，`landing_progress` 的 active_rate >50% 且 episode_sum_mean 显著高于 0。
8. `main_risk`：乘积因子在过渡阶段可能塌缩为 0 导致新增信号失效，若同时 contact_reward 仍主导且无速度/角度惩罚强化减速，agent 可能继续满足于低 contact 徘徊而不进入高稳定区，从而新组件不具备足够的梯度引导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    angvel = obs[5]
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # 1. Progress towards center (0,0)
    dist = (x**2 + y**2)**0.5 + 1e-6
    next_dist = (next_x**2 + next_y**2)**0.5 + 1e-6
    progress_delta = dist - next_dist

    # 2. Orientation stability penalty (hinge)
    angle_threshold = 0.3
    angvel_threshold = 0.5
    angle_violation = max(0.0, abs(next_angle) - angle_threshold)
    angvel_violation = max(0.0, abs(next_angvel) - angvel_threshold)
    orientation_penalty = -0.1 * angle_violation - 0.05 * angvel_violation

    # 3. Speed safety penalty (hinge)
    speed_threshold = 0.5
    vx_violation = max(0.0, abs(next_vx) - speed_threshold)
    vy_violation = max(0.0, abs(next_vy) - speed_threshold)
    speed_penalty = -0.05 * (vx_violation + vy_violation)

    # 4. Contact encouragement (dense reward for feet on ground)
    contact_reward = 0.1 * (next_left + next_right)

    # 5. NEW: Landing progress reward -- drives completion and stabilization
    # Proximity to center: 1.0 when at center, falls to 0 at distance 0.5
    proximity = max(0.0, 1.0 - next_dist / 0.5)
    # Velocity moderation: 1.0 when total speed near zero, falls off
    velocity_moderation = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / 0.5)
    # Orientation stability: 1.0 when angle near zero
    orientation_stability = max(0.0, 1.0 - abs(next_angle) / 0.2)
    # Angular velocity moderation
    angvel_moderation = max(0.0, 1.0 - abs(next_angvel) / 0.3)
    # Contact factor: continuous [0,1] for average contact
    contact_factor = (next_left + next_right) / 2.0

    landing_factor = proximity * velocity_moderation * orientation_stability * angvel_moderation * contact_factor
    landing_progress = 0.15 * landing_factor

    total_reward = progress_delta + orientation_penalty + speed_penalty + contact_reward + landing_progress

    components = {
        'progress_delta': progress_delta,
        'orientation_penalty': orientation_penalty,
        'speed_penalty': speed_penalty,
        'contact_reward': contact_reward,
        'landing_progress': landing_progress
    }

    return float(total_reward), components
```
