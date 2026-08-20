# Response Record

## 1. evidence
当前最终策略得分27.34（目标200），episode平均长度800.4，50%截断/50%终止；contact_reward以92.9%的magnitude_share绝对主导（episode_sum_mean=202.4），active_rate仅4.6%但每次激活累积巨大；goal_proximity因势能差分形式仅贡献9.19均值（4.2% signed_share）；外部得分远低于内部奖励合计（~207），呈典型proxy hacking；上一轮将goal_proximity从状态惩罚改为势能差分并提高w_contact从2.0到8.0，结果恶化。

## 2. behavior_diagnosis
agent在约50%回合中无法到达目标（截断），到达时也非稳定软着陆——contact_reward的巨大持久收益使agent一旦获得双腿接触就持续收割奖励，而非追求真正的着陆质量；外部评分远低于proxy合计，确认agent在最大化错误的奖励结构。

## 3. signal_completeness
goal_proximity（best版的状态距离惩罚）、velocity_penalty、orientation_penalty职责基本完备且数学形态合理；contact_reward的持久状态奖励形态是问题根源——占据好状态即可持续获奖（state_to_improvement模式），缺乏着陆质量门控和一次性完成信号。

## 4. selected_level
**Level 2**：contact_reward的证据模式匹配`persistent_to_transition_event`（持续状态→有效状态转移），且搜索结果确认contact_reward_hacking症状——contact奖励高但外部得分低，应将contact_reward从持久状态奖励转变为带质量门控的一次性着陆事件奖励。

## 5. selected_intervention
以best代码为基础，将contact_reward从`w_contact * both_legs_contact * proximity`（每步持久奖励）改为一次性转移奖励：`w_contact * landing_transition * proximity_next * speed_quality * angle_quality`，其中landing_transition检测双腿接触的上升沿，speed_quality和angle_quality用连续bounded因子门控着陆质量，w_contact同步调整为50.0以匹配一次性事件的量级。

## 6. falsifiable_hypothesis
持久contact_reward允许agent在粗猛着陆后持续收割奖励（当前proxy≈207 vs 外部27），改为一次性转移奖励+质量门控后，agent将无法通过维持不良着陆状态获利，必须追求真正的低速、竖直、稳定着陆才能获得高额bonus，proxy与外部得分应重新对齐。

## 7. expected_next_round
contact_reward的episode_sum_mean应大幅下降（从202降至个位数到几十，取决于着陆质量），magnitude_share从92.9%显著回落；goal_proximity的相对份额上升；外部得分应提高（目标朝200移动），terminated比例中高质量着陆增加；active_rate可能进一步下降但不再是问题因为改为一次性事件。

## 8. main_risk
一次性bonus的稀疏性可能使学习变慢——若active_rate降至接近0%，agent将失去着陆相关反馈；连续质量因子（speed_quality × angle_quality）若阈值不当可能导致bonus过小，需下一轮验证系数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    x = obs[0]            # horizontal position relative to target pad
    y = obs[1]            # vertical position relative to target pad
    vx = obs[2]           # horizontal velocity
    vy = obs[3]           # vertical velocity
    angle = obs[4]        # body angle
    ang_vel = obs[5]      # angular velocity
    left_contact = obs[6] # left support leg contact (0 or 1)
    right_contact = obs[7]# right support leg contact (0 or 1)

    # next_obs
    nx = next_obs[0]
    ny = next_obs[1]
    nvx = next_obs[2]
    nvy = next_obs[3]
    n_angle = next_obs[4]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # Hyperparameters
    w_goal = 1.0
    alpha_proximity = 5.0
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 50.0          # raised for one-time landing bonus
    beta_speed = 10.0          # speed quality decay: 1/(1+beta_speed*speed_sq)
    beta_angle = 10.0          # angle quality decay: 1/(1+beta_angle*angle_sq)

    # Distance to target center (squared) for current state
    dist_sq = x**2 + y**2

    # Soft proximity weight for velocity penalty (uses current position)
    proximity = 1.0 / (1.0 + alpha_proximity * dist_sq)

    # 1. Main progress: drive toward target center (dense quadratic penalty on distance)
    goal_proximity = -w_goal * dist_sq

    # 2. Soft landing velocity penalty: active only near the target
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: one-time landing bonus on transition to dual-leg contact,
    #    gated by proximity, speed quality, and angle quality at the moment of landing.
    obs_dual = left_contact * right_contact
    next_dual = n_left_contact * n_right_contact

    # Rising edge: newly achieved dual-leg contact (0->1 transition)
    landing_transition = max(0.0, next_dual - obs_dual)

    # Quality gates based on next_obs (state at landing moment)
    dist_sq_next = nx**2 + ny**2
    proximity_next = 1.0 / (1.0 + alpha_proximity * dist_sq_next)
    speed_quality = 1.0 / (1.0 + beta_speed * (nvx**2 + nvy**2))
    angle_quality = 1.0 / (1.0 + beta_angle * (n_angle**2))

    contact_reward = w_contact * landing_transition * proximity_next * speed_quality * angle_quality

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components
```
