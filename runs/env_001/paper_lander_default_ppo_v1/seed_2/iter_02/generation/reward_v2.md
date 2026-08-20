`evidence`：所有episode均提前终止（terminated=20/20），score中位数约-114，episode_length仅68.4。distance_reward占magnitude 93%且始终为负（-2.0*dist），soft_landing_proxy active_rate仅0.3%，几乎没有正向信号。历史记忆显示这是第一轮，无前序干预。

`behavior_diagnosis`：agent在~68步内坠毁或出界，累积大量负距离代价后提前失败。软着陆条件极难同时满足，0.3%触发率意味着agent从未有效收到完成信号。

`signal_completeness`：缺少正向进展信号——全程只有惩罚（距离+稳定性），没有"正在接近目标"的正反馈。着陆信号虽存在形式但实质上不可达（active_rate 0.3%）。任务所需的"接近→减速→稳定→着陆"梯度链中，第一环就塌缩为纯惩罚。

`selected_level`：Level 2，触发条件符合`state_to_improvement`证据模式——占据好状态（接近目标）只能减少惩罚而非获得正奖励，且当前纯惩罚结构下策略快速失败。同时符合`early_failure_or_crash`症状（score<-100, len<150, 提前终止100%）。

`selected_intervention`：将`distance_reward`从状态值`-2.0 * dist_to_target`改为进展量`2.0 * (prev_dist - curr_dist)`，保持系数2.0不变。其余组件（stability_penalty、soft_landing_proxy）完全不改动。

`falsifiable_hypothesis`：进展增量提供明确的"靠近=正奖励"信号，消除"无论如何都受罚"的困境。episode级累计从无界负值变为有界值（2.0×初始距离），使着陆奖励和稳定性惩罚在相对尺度上变得有意义，agent应能学会接近目标而非随机坠毁。

`expected_next_round`：episode_length应上升（减少早期坠毁），terminated比例可能仍高但score应显著改善（远离-114），distance_reward的episode_sum_mean应从-133变为接近0或正值的有限值，soft_landing_proxy的active_rate可能略有上升（agent更常到达目标附近）。

`main_risk`：纯进展信号在到达目标后归零，若着陆奖励仍不可达，agent可能在目标附近徘徊或震荡而不完成着陆——形成"中等分平台"。下一轮若出现此症状需对landing_proxy做sparse_to_dense变换。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next observation variables
    prev_x, prev_y = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress signal: reward getting closer to target, penalize moving away
    prev_dist = (prev_x**2 + prev_y**2) ** 0.5
    curr_dist = (x**2 + y**2) ** 0.5
    reward_dist = 2.0 * (prev_dist - curr_dist)

    # 2. Stability constraint: penalise high speeds and large angles
    reward_stability = (
        -0.1 * abs(vx) -
        0.1 * abs(vy) -
        0.1 * abs(angle) -
        0.1 * abs(omega)
    )

    # 3. Soft landing proxy: reward simultaneous near-target, low-speed,
    #    upright attitude and both legs contacting the pad.
    prox_dist_thresh = 0.3
    prox_vel_thresh = 0.2
    prox_angle_thresh = 0.1

    condition = (
        curr_dist < prox_dist_thresh and
        abs(vx) < prox_vel_thresh and
        abs(vy) < prox_vel_thresh and
        abs(angle) < prox_angle_thresh and
        left_contact > 0.5 and
        right_contact > 0.5
    )
    reward_landing = 1.0 if condition else 0.0

    total_reward = reward_dist + reward_stability + reward_landing

    components = {
        "distance_progress": reward_dist,
        "stability_penalty": reward_stability,
        "soft_landing_proxy": reward_landing
    }

    return float(total_reward), components
```