**1. evidence**：当前得分-13.71，外部远低于target=200，19/20集被截断(995步)，仅1集自然终止；landing_proxy每集均值2021.8、占magnitude_share 82%、active_rate 100%，而distance_penalty仅-426.9——代理在无接触悬停状态下持续收割proxy奖励，从未真正settle。

**2. behavior_diagnosis**：策略学会了在目标平台上方悬停——保持低速度、竖直姿态、靠近原点，但不触地（或仅间歇触地），从而每步持续收取~2点landing_proxy，直至1000步截断。外部任务显然惩罚悬停耗时，导致外部分数为负。

**3. signal_completeness**：distance_penalty（位置接近）、velocity_penalty（近目标减速）、orientation_penalty（姿态稳定）职责完备且合理。landing_proxy本应承担safe_landing_confirmation，但其“占据好状态即可持续获奖”的数学形态导致proxy悬停漏洞——缺失的是完成对齐信号（settle后终止，代理反而失去持续收益）。

**4. selected_level**：Level 2——`proxy_to_completion_alignment`。证据模式完全匹配“proxy提高但外部任务不升”：内部proxy每集2000+而外部得分-13.71。接触门控是最小结构变换，强制“触地才能领奖”，消除悬停漏洞。

**5. selected_intervention**：唯一修改`landing_proxy`组件——将原来始终激活的加权和乘以接触门控`max(left_contact, right_contact)`，使无接触时landing_proxy恒为0。权重保持w_landing=3.0。其他三个组件不变。

**6. falsifiable_hypothesis**：接触门控后，agent在飞行下降阶段landing_proxy=0，只有distance_penalty的负向引导，必须触地才能获得正向奖励。这将迫使agent学习触地→双支撑接触→最终settle，而非无限悬停。若假说成立，应看到terminated比例上升、episode_length下降。

**7. expected_next_round**：landing_proxy的episode_sum_mean将大幅下降（飞行段无收益），signed_share下降；terminated比例上升；外部score应改善（少截断、更早完成）；active_rate可能从100%下降（仅触地时激活）。

**8. main_risk**：飞行下降阶段只有负信号，早期探索中agent可能发现不了触地收益（credit assignment距离过长），导致收敛慢或退化为随机坠落。若下一轮terminated上升但score仍差，说明需要补充飞行阶段的轻度引导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D vehicle landing task.

    Components:
    - distance_penalty: encourages moving towards the target platform (position=0,0)
    - velocity_penalty: penalizes high speed when near target, gated by distance
    - orientation_penalty: penalizes tilt and angular velocity
    - landing_proxy: contact-gated sum of bounded factors rewarding simultaneous
      proximity, low speed, upright posture, and ground contact;
      ONLY active when at least one support touches the ground
    """
    # next_obs unpacking
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Euclidean distance to target
    dist = (x**2 + y**2) ** 0.5

    # Component A: distance penalty (core progress signal)
    w_dist = 1.0
    distance_penalty = -w_dist * dist

    # Component B: velocity penalty (damped by distance to target)
    w_vel = 0.2
    gate = 1.0 / (1.0 + dist)
    speed_sq = vx**2 + vy**2
    velocity_penalty = -w_vel * speed_sq * gate

    # Component C: orientation stabilization penalty
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * abs(angle) - w_angvel * abs(ang_vel)

    # Component D: contact-gated landing proxy
    # Only active when at least one support touches the ground,
    # preventing the agent from farming reward by hovering above the platform
    w_landing = 3.0
    # proximity factor: 1 at dist=0, 0 when dist >= 2.5
    prox_reward = max(0.0, 1.0 - dist / 2.5)
    # speed factor: 1 at speed=0, 0 when speed >= 2.0 m/s
    speed = (vx**2 + vy**2) ** 0.5
    vel_reward = max(0.0, 1.0 - speed / 2.0)
    # angle factor: 1 at angle=0, 0 when |angle| >= 0.5 rad
    angle_reward = max(0.0, 1.0 - abs(angle) / 0.5)
    # contact factor: 0, 0.5, or 1.0 depending on single/double contact
    contact_reward = (left_contact + right_contact) / 2.0
    # contact gate: 0 when no contact, 1 when any contact exists
    contact_gate = max(left_contact, right_contact)
    # average of four bounded factors, gated by ground contact
    landing_proxy = w_landing * contact_gate * (prox_reward + vel_reward + angle_reward + contact_reward) / 4.0

    # Total reward
    total_reward = distance_penalty + velocity_penalty + orientation_penalty + landing_proxy

    # Component dictionary
    components = {
        "distance_penalty": distance_penalty,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components
```