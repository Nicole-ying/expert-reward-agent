# Response Record

1. `evidence`：v2 (score 217) 是历史最佳，v3 删除零贡献的 crash 组件后分数降至 212.9、terminated 从 18 降至 16、len 从 484 升至 512，表明当前骨架能力未变但训练波动略差，当前 landing_success_reward 主导（per-step 0.073）、living_penalty 恒为 -0.002 (per-step) 且被淹没。
2. `behavior_diagnosis`：策略能稳定着陆成功 (terminated 16/20)，但有 4 回合因超时或未满足条件而失败，部分回合耗时较长，表明着陆过程存在效率优化空间。
3. `signal_completeness`：goal_proximity 提供距离方向、landing_success_reward 提供最终着陆正信号、living_penalty 施加时间压力，三种职责完备、可达。
4. `selected_level`：Level 1（调系数），因为 per-step 量级显示 living_penalty 过于微弱（绝对值 0.002 vs landing 0.073），调整幅度可加强时间激励而不改变数学形态。
5. `selected_intervention`：唯一目标组件为 living_penalty，将其系数从 -0.002 增强至 -0.005。
6. `falsifiable_hypothesis`：适度增大时间惩罚会驱动策略更快完成着陆，减少徘徊步数，进而提高 terminated 比例并提升奖励总和。
7. `expected_next_round`：score 应小幅上升或持平，episode_length 下降，terminated 数从 16 增加到 18 或以上。
8. `main_risk`：惩罚过强可能导致早期策略因步数积累惩罚而压过低频的 landing 奖励，从而引发更快但更易失败的激进行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离计算
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（距离改善差分）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)

    # 2. 着陆成功奖励
    landing_dist_thresh = 0.2
    landing_vel_thresh = 0.15
    landing_ang_thresh = 0.1
    w_landing = 1.0

    contacts = left_contact + right_contact
    near_center = next_dist < landing_dist_thresh
    low_speed = abs(next_vx) < landing_vel_thresh and abs(next_vy) < landing_vel_thresh
    upright = abs(body_angle) < landing_ang_thresh

    landing_success = (contacts == 2) and near_center and low_speed and upright
    reward_landing = w_landing if landing_success else 0.0

    # 3. 时间惩罚（从 -0.002 加强至 -0.005）
    living_penalty = -0.005

    total_reward = reward_prox + reward_landing + living_penalty

    components = {
        "goal_proximity": reward_prox,
        "landing_success_reward": reward_landing,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```
