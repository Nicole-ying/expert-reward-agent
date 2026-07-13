# Response Record

## 1. 行为与历史诊断

`evidence`：当前score=206.14已超target(200)，但safe_contact_reward episode_sum_mean=294.2占signed_share的94.6%，active_rate=51.4%意味着约350步在双腿触地状态累积奖励；progress_reward仅2.77(0.9%)几乎无引导力。terminated=10/20、truncated=10/20、平均episode长度681步。

`behavior_diagnosis`：智能体已学会飞向着陆垫并双腿着陆，但随后利用safe_contact_reward的持续发放"刷分"——保持着陆姿态收集逐步奖励，直到settled终止或时间截断，而非尽快完成着陆后终止。10个truncated episode说明有一半情况刷到时间上限仍未settle。

`signal_completeness`：职责基本完备——progress提供到达引导、stability/angle约束姿态、fuel惩罚浪费、contact标识着陆事件。但contact信号被设为持续状态奖励而非一次性完成事件，导致刷分行为。

`selected_level`：**Level 2** — persistent_to_transition_event。证据明确：safe_contact_reward的94.6%占比来自持续状态累积（active_rate 51.4% × 350步 ≈ 持续刷分），而非一次性完成signal。这是典型的"占据好状态即可持续获奖"模式，需将持续性状态奖励变为状态转移事件奖励。

`selected_intervention`：唯一目标是**safe_contact_reward**，从每步持续的`contact_now * quality_factors`变换为一次性着陆转移事件`just_landed * landing_quality_factors`，系数从1.0提升至100.0以匹配一次性事件的尺度需求。

`falsifiable_hypothesis`：将safe_contact_reward从持续状态奖变为转移事件奖后，agent不再有动机在着陆垫上停留刷分，episode应更快以settled终止；一次性的100点量级着陆bonus应保持足够强信号让agent学会着陆；progress_reward+stability_penalty提供到达过程的局部引导。

`expected_next_round`：safe_contact_reward的episode_sum_mean应降至~50-100范围（一次性事件），active_rate应骤降（仅在着陆转移步触发），terminated比例应上升（不再刷分等待超时），episode_length应显著下降，score可能短期下降但仍应接近或超过target。

`main_risk`：转移事件比持续状态稀疏得多，信用分配变难——agent可能在早期训练中更难发现着陆行为与奖励的关联，导致score暂时下降。若着陆质量因子gate过多（dist/vel/angle三重门控），稀疏性可能被放大。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact_prev = obs[6]
    right_contact_prev = obs[7]

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    next_left = next_obs[6]
    next_right = next_obs[7]

    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # ------------------ 1. Progress reward ------------------
    w_progress = 2.0
    progress = dist - next_dist
    progress_reward = w_progress * progress

    # ------------------ 2. Stability penalty (gated) ------------------
    gate = 1.0 / (1.0 + dist)
    w_vel = 0.2
    w_ang = 0.2
    stability_error = w_vel * (vx ** 2 + vy ** 2) + w_ang * (angle ** 2 + angvel ** 2)
    stability_penalty = -gate * stability_error

    # ------------------ 3. Fuel penalty ------------------
    w_fuel = 0.02
    fuel_penalty = -w_fuel if action != 0 else 0.0

    # ------------------ 4. Safe landing transition bonus ------------------
    # Detect first-ever both-legs contact (binary transition 0->1)
    prev_both_contact = left_contact_prev * right_contact_prev
    curr_both_contact = next_left * next_right
    just_landed = 1.0 if (curr_both_contact > 0.5 and prev_both_contact < 0.5) else 0.0

    # Landing quality gates at the moment of first contact
    landing_dist_factor = 1.0 / (1.0 + 10.0 * next_dist)
    landing_vel_factor = 1.0 / (1.0 + 5.0 * (next_vx ** 2 + next_vy ** 2))
    landing_angle_factor = 1.0 / (1.0 + 5.0 * (abs(next_angle) + abs(next_angvel)))

    w_landing = 100.0
    safe_contact_reward = w_landing * just_landed * landing_dist_factor * landing_vel_factor * landing_angle_factor

    total_reward = progress_reward + stability_penalty + fuel_penalty + safe_contact_reward

    reward_components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "fuel_penalty": fuel_penalty,
        "safe_contact_reward": safe_contact_reward
    }

    return float(total_reward), reward_components
```
