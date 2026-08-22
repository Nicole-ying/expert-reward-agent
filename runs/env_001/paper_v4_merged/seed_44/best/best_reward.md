1. evidence：terminated 20/20, len=506.35, contact_reward 5.8% active yet dominates signed share (87.5%), while progress (99.8% active, 0.03 mean) and landing_reward are negligible. Previous iteration added contact_reward=5.0 and score collapsed from 144.30 to 2.30.
2. behavior_diagnosis：Agent likely exploits sparse 5.0 contact bonus to early‑terminate (crash or rough landing), ignoring the continuous progress and stability signals needed for a soft, controlled landing.
3. signal_completeness：All obs dimensions are used; the missing element is reliable completion credit without dominating the reward—current contact bonus is too sparse and too large.
4. selected_level：Level 2 (structural change) – the contact_reward added in the previous round caused a score collapse, and the sparse large bonus creates a proxy misalignment (sparse→dense transition is needed; see sparse→dense evidence pattern). Removing it is the minimal reversible intervention.
5. selected_intervention：Set contact_reward = 0.0 (effectively remove the sparse contact bonus component) and keep all other components unchanged.
6. falsifiable_hypothesis：Without the 5.0 sparse contact bonus, the agent will stop the premature termination pattern and return to the stable progress‑driven behaviour seen in iteration 2, raising episode length and score.
7. expected_next_round：Episode length increases to ~900‑1000, terminated rate should drop (more truncations), score rises above 100, contact_reward share plunges to 0%, and progress magnitude_share becomes the dominant signal.
8. main_risk：Without a completion bonus the agent may learn to hover indefinitely rather than land, leaving a gap to the 200‑target that will need a soft‑completion reward later.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 距离进展
    x_curr = obs[0]
    y_curr = obs[1]
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5

    x_next = next_obs[0]
    y_next = next_obs[1]
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    progress = dist_curr - dist_next

    # 姿态与角速度惩罚（保持低摇晃）
    body_angle_next = next_obs[4]
    angular_vel_next = next_obs[5]
    angle_penalty = body_angle_next ** 2 + angular_vel_next ** 2

    # 连续软着陆引导
    dist_factor = 2.718281828 ** (-dist_next / 0.5)
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    speed_factor = max(0.0, 1.0 - (abs(x_vel_next) + abs(y_vel_next)) / 1.0)
    landing_reward = dist_factor * speed_factor

    # 移除主导的稀疏接触奖励
    contact_reward = 0.0

    total = (
        10.0 * progress
        - 0.5 * angle_penalty
        + 0.01 * landing_reward
        + contact_reward
    )

    components = {
        "progress": 10.0 * progress,
        "angle_penalty": -0.5 * angle_penalty,
        "landing_reward": 0.01 * landing_reward,
        "contact_reward": contact_reward
    }

    return float(total), components
```