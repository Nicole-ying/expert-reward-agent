`evidence`: iter 2 (current) score=-129, len=70, 17/20 early terminal; progress dominates (95.9% mag) and rewards any distance decrease including crashing; landing_reward active_rate仅3.7%且产品形式塌缩；iter 1 (best) score=-70, len=1000，proximity让agent存活但未成功着陆。
`behavior_diagnosis`: iter 2 agent学会快速坠向目标以获取progress奖励，导致高速撞击地面提前终止（平均70步）；iter 1 agent在proximity引导下存活完整episode但滞留在目标附近不完成着陆。
`signal_completeness`: 缺少可到达的着陆信号——原landing_reward三因子乘积(contact×speed×angle)使偏导同时接近零的样本占绝大多数，active_rate 3.7%且magnitude_share仅1.9%，无法为软着陆提供有效学习梯度；proximity本身无法区分悬停和着陆完成。
`selected_level`: Level 2，触发条件为`product_to_noncollapsing_joint`证据——多因子乘积导致奖励几乎恒为零，即使个别因子偶尔改善也得不到反馈。
`selected_intervention`: 以best代码（proximity + landing_reward + fuel_penalty）为基础，单独将landing_reward从乘积形式`contact_factor * speed_factor * angle_factor`改为加性非塌缩形式`contact_score + speed_score + angle_score`，各因子用bounded线性衰减独立贡献。
`falsifiable_hypothesis`: 加性着陆奖励使agent在接近地面时获得关于接触、速度和姿态的独立梯度，即使某些维度未达标也能收到部分反馈；应提高landing_reward的active_rate和episode_sum_mean，且agent应能学习到可控下降而非撞击。
`expected_next_round`: landing_reward的active_rate应上升至15%以上，episode_sum_mean应上升，episode length应不再以early terminal为主，score应相对iter 1的-70有所改善。
`main_risk`: 加性形式可能让agent通过部分达标（如低速但未接触）获取中等奖励而不完成完整着陆；高度门控`y < 0.5`可限制作用范围，但若阈值过高可能延长悬停行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs for state-after-action evaluation
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # 1. Proximity: state-based bounded reward for being near target center
    dist = (next_x ** 2 + next_y ** 2) ** 0.5
    proximity = 0.1 / (1.0 + dist)

    # 2. Landing reward: product_to_noncollapsing_joint transform
    #    Old: contact_factor * speed_factor * angle_factor (collapses to ~0)
    #    New: additive independent contributions, gated by height
    threshold_y = 0.5
    if next_y < threshold_y:
        # Contact: each leg independently contributes, range [0, 2]
        contact_score = next_left_contact + next_right_contact
        # Speed: bounded linear decay, 1 at rest, 0 when total speed >= 2.0
        total_speed = abs(next_x_vel) + abs(next_y_vel)
        speed_score = max(0.0, 1.0 - total_speed / 2.0)
        # Angle: bounded linear decay, 1 when upright, 0 when |angle| >= 0.5
        angle_score = max(0.0, 1.0 - abs(next_angle) / 0.5)
        landing_reward = 1.0 * (contact_score + speed_score + angle_score)
    else:
        landing_reward = 0.0

    # 3. Fuel efficiency penalty
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = proximity + landing_reward + fuel_penalty

    components = {
        "proximity": proximity,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```