```json
{
  "evidence": "Subagent报告显示contact_completion占比84.8%但active_rate仅15.8%，每步均值0.538远高于approach_progress的0.0066；8/20成功终止，12/20超时，说明着陆奖励过于稀疏。",
  "behavior_diagnosis": "策略已经学会靠近平台，但因双足同时接地的乘积门控过于苛刻，多数回合在平台附近低速徘徊，最终因无法触发settle条件而超时。",
  "signal_completeness": "当前奖励缺少从接近到成功着陆的中间引导，contact_completion的稀疏性使agent难以可靠学习最后阶段的精确动作。",
  "selected_level": "Level 2，将contact_completion的公式由稀疏乘积改为加性，以提供单脚接地的连续奖励，并调整权重避免奖励爆炸。",
  "selected_intervention": "唯一修改contact_completion组件：将`left*right`替换为`left+right`，同时将w_contact从5.0降至2.5，保持距离/速度门控不变。",
  "falsifiable_hypothesis": "单脚接地的连续奖励能够引导agent逐步完成两脚着陆，提高成功终止率、缩短超时，从而提升score。",
  "expected_next_round": "terminated比率应升至>10/20，score>120，episode_length可能缩短，contact_completion的active_rate升至>30%且magnitude_share适度下降。",
  "main_risk": "agent可能学会维持单脚接触而不稳定另一脚，导致更多超时或姿态异常；需警惕orientation和速度约束是否被削弱。"
}
```

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 1. Target approach: improvement delta ----------
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = dist_curr - dist_next           # positive when agent moves closer
    w_approach = 5.0
    reward_approach = w_approach * progress

    # ---------- 2. Soft landing speed: distance-gated velocity penalty ----------
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    sigma = 0.5                                # determines how sharply gate rises near target
    gate = 2.718281828 ** (-dist_next**2 / sigma**2)
    w_speed = 2.0
    reward_speed = -w_speed * speed_next * gate

    # ---------- 3. Stable orientation: quadratic penalties on tilt and rotation ----------
    w_angle = 1.0
    w_angvel = 0.1
    reward_orientation = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # ---------- 4. Contact completion: dual-leg contact gated by position and speed ----------
    left = left_contact_next
    right = right_contact_next
    beta_dist = 10.0
    beta_speed = 10.0
    # Soft proximity factors in [0,1] – near 1 only when distance/speed are near zero
    dist_factor = 1.0 / (1.0 + beta_dist * dist_next**2)
    speed_factor = 1.0 / (1.0 + beta_speed * speed_next**2)
    # Modified: additive contact to give partial reward for single-leg touchdowns
    w_contact = 2.5
    reward_contact = w_contact * (left + right) * dist_factor * speed_factor

    total = reward_approach + reward_speed + reward_orientation + reward_contact

    components = {
        'approach_progress': reward_approach,
        'soft_landing_speed_penalty': reward_speed,
        'orientation_penalty': reward_orientation,
        'contact_completion': reward_contact,
    }
    return float(total), components
```