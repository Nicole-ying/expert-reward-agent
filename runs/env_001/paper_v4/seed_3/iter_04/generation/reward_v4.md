1. **evidence**：当前 `settling_quality` 的 `active_rate=0.0%`，episode 全部 truncated（20/20）无 terminated，score=-15.42 远低于 best 的 147.71；上一轮将 `contact_bonus`（best 中 per-step≈0.241、主导学习）替换为需双腿接触+近目标+低速+低姿态角的联合条件，导致着陆信号完全不可达。
2. **behavior_diagnosis**：agent 失去了所有接触/着陆引导，仅靠 progress 到达目标附近但从不尝试放腿着陆，全程滑行存活 1000 步直到 truncation，没有一次成功 settled termination。
3. **signal_completeness**：progress（过程引导）、velocity_penalty（近目标减速）、orientation_penalty（姿态稳定）职责完备且可达；但着陆/接触职责的当前数学形态（联合硬条件）使 agent 永远得不到该信号的梯度，属于信号不可达。
4. **selected_level**：Level 2，匹配 `sparse_to_dense` 模式 — `settling_quality` 的 `active_rate=0%` 远低于 ~1% 的可学习阈值，需将硬性联合条件拆解为可达的连续接触梯度 + 质量加成。
5. **selected_intervention**：唯一修改 `settling_quality` → `landing_reward`，拆为两部分：(a) `contact_reward` — 近目标时任意腿接触即给连续梯度（0/1/2 级）；(b) `settling_bonus` — 双腿同时接触且低速度低姿态时额外加成。保持 proximity_gate 防止远离目标刷接触。
6. **falsifiable_hypothesis**：给 agent 一个可达的接触梯度（单腿→双腿）后，它应能在 progress 引导下到达目标附近并获得接触反馈，从而学会放腿着陆；`landing_reward` 的 `active_rate` 应明显 >0，terminated 比例应上升。
7. **expected_next_round**：`landing_reward` 的 `active_rate` 从 0% 升至显著正值，terminated 集数 >0，score 明显回升且不应再为负值。
8. **main_risk**：`contact_reward` 可能让 agent 学会在目标附近反复轻点单腿而不真正稳定着陆（contact hacking），但 `settling_bonus` 和现有 velocity_penalty 提供了对抗梯度；若下一轮 `landing_reward` 活跃但 terminated 仍为 0，则需要进一步收紧着陆质量条件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function with progress guidance and reachable landing gradient.

    Components:
    - progress: rewards reducing distance to target center (0,0)
    - velocity_penalty: speed penalty gated by proximity to target
    - orientation_penalty: penalty for tilt and angular velocity
    - landing_reward: proximity-gated contact gradient + settling quality bonus
      Replaces the unreachable settling_quality with a staged design:
      (a) contact_reward provides accessible gradient for any leg contact near target
      (b) settling_bonus provides aspirational reward for dual-contact quality landing
    """
    # Current state
    x_pos, y_pos = obs[0], obs[1]
    # Next state
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distances to target
    prev_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # 1. Progress reward: positive for moving toward target, negative for moving away
    progress_weight = 5.0
    progress_reward = progress_weight * (prev_dist - next_dist)

    # 2. Velocity penalty – active only when close to target
    proximity_factor = 1.0 / (1.0 + 10.0 * next_dist)
    vel_weight = 1.0
    velocity_penalty = -vel_weight * proximity_factor * (nx_vel ** 2 + ny_vel ** 2)

    # 3. Orientation stability – keep body upright and avoid spinning
    orient_weight = 0.5
    orientation_penalty = -orient_weight * (n_angle ** 2 + 0.2 * n_angvel ** 2)

    # 4. Landing reward – staged: accessible contact gradient + quality bonus
    #    contact_score provides continuous gradient: 0 (no legs) / 1 (one leg) / 2 (both legs)
    contact_score = left_contact + right_contact
    contact_weight = 0.3
    contact_reward = contact_weight * proximity_factor * contact_score

    #    settling_bonus: extra reward when both legs contact with low velocity and stable attitude
    vel_magnitude = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    settling_factor = 1.0 / (1.0 + 5.0 * vel_magnitude + 2.0 * abs(n_angle) + 1.0 * abs(n_angvel))
    contact_product = left_contact * right_contact
    settling_bonus_weight = 0.5
    settling_bonus = settling_bonus_weight * contact_product * proximity_factor * settling_factor

    landing_reward = contact_reward + settling_bonus

    total_reward = progress_reward + velocity_penalty + orientation_penalty + landing_reward

    components = {
        'progress': float(progress_reward),
        'velocity_penalty': float(velocity_penalty),
        'orientation_penalty': float(orientation_penalty),
        'landing_reward': float(landing_reward)
    }

    return float(total_reward), components
```