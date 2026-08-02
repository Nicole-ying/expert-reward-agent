1. `evidence`：全量episode truncated（len=1000），无terminated失败；progress与landing_reward主导增益，speed_penalty活跃率0.1%为僵尸组件；obs[6]、obs[7]双腿接触未使用；score 144.3距target 55.7。
2. `behavior_diagnosis`：agent已学会靠近目标垫并减速，但未能触发着陆，形成悬停至超时的策略；由于缺乏接触反馈，无法诱导已具备条件的成功settle。
3. `signal_completeness`：缺少着陆完成激励，现有观测中left/right support contact未被奖励函数利用，该信号对触发success-like终止是必要的。
4. `selected_level`：Level 2，触发条件：僵尸组件speed_penalty（活跃率<2%）和无接触观测导致信号缺口，需删除僵尸组件并添加新组件。
5. `selected_intervention`：删除speed_penalty，新增contact_reward组件，以5.0系数奖励双腿同时接触（next_obs[6]*next_obs[7]）的每步，并保留progress、landing_reward和angle_penalty。
6. `falsifiable_hypothesis`：添加接触奖励后，接近目标且双腿触地的状态将获得显著正反馈，引导策略主动触发着陆并提前终止，从而提升score并降低mean episode length。
7. `expected_next_round`：contact_reward的active_rate>0%，mean episode length从1000下降，出现部分提前终止，score≥150；progress与landing_reward仍为主力，angle_penalty继续抑制摇晃。
8. `main_risk`：接触奖励可能诱导agent在较高速度下强行触垫，引发crash类终止，若下轮出现terminated失败需追加速度约束。

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

    # 新增：双腿接触完成奖励
    contact_both = next_obs[6] * next_obs[7]   # 0 或 1
    contact_reward = contact_both * 5.0

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