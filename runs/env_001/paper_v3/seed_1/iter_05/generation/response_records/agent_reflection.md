# Response Record

`evidence`: 当前得分-116.98，19/20回合在68步内提前终止（crash），目标接近度均值dist_sq≈1.13说明agent能接近目标但无法软着陆；contact_reward的active_rate仅1%、episode_sum_mean仅1.59，过渡型着陆奖励几乎不触发，sparse_completion_proxy启发式直接命中此症状；上一轮改动（transition-based + quality gates, w_contact=50）相比best（continuous, w_contact=2.0）使active_rate塌缩且得分严重恶化。

`behavior_diagnosis`: Agent能到达目标附近（平均距离约1.06单位），但在着陆瞬间失控——高速撞击或姿态失稳导致crash终止。接触奖励过于稀疏（1%触发率），agent在关键着陆阶段得不到可学习的反馈，只能依赖goal_proximity冲向目标后崩溃。

`signal_completeness`: goal_proximity（稠密方向引导）、velocity_penalty（近距减速）、orientation_penalty（姿态稳定）职责基本完备且可达；但contact_reward的关键着陆反馈因过渡型+品质门控塌缩为稀疏事件（active_rate=1%），在agent最需要学习软着陆时信号缺失。

`selected_level`: Level 2 — sparse_to_dense变换。证据直接匹配"sparse_completion_proxy"模式（active_rate≤1%无法提供可达引导），且best的continuous版本曾获得更高分，证明连续形态对该任务是可达的。

`selected_intervention`: 唯一目标组件为contact_reward，从"过渡型+品质门控乘积"改为"连续接触×近距门控×速度品质衰减"。具体：用`both_legs_contact * proximity`替代`landing_transition * proximity_next`使奖励持续化，保留`speed_quality = 1/(1+beta_speed*(vx²+vy²))`作为速度调制但移除angle_quality（orientation_penalty独立处理角度），w_contact从50降至5以适应连续累积形态。

`falsifiable_hypothesis`: 连续接触奖励使active_rate从1%显著提升，agent在近距双足触地时获得即时正反馈，从而学会在接近目标后减速并保持双足接触而非冲撞crash；episode_length应延长、early_terminal比例应下降、score应改善。

`expected_next_round`: contact_reward的active_rate应从~1%升至明显更高（预计>20%），episode_sum_mean应变为正且显著增大；score应脱离-116严重负值区间；early_terminal(<150步且score<-50)回合数应从19/20下降。

`main_risk`: 连续接触奖励可能被agent利用——在目标垫上轻微弹跳或晃动维持双足接触来持续累积奖励，而不真正settle。速度品质衰减可部分缓解，但若出现该漏洞需在后续轮次加入settlement事件型bonus区分"接触中"与"已稳定停靠"。

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

    # Hyperparameters
    w_goal = 1.0
    alpha_proximity = 5.0
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 5.0           # moderate weight for continuous contact bonus
    beta_speed = 10.0          # speed quality decay: 1/(1+beta_speed*speed_sq)

    # Distance to target center (squared) for current state
    dist_sq = x**2 + y**2

    # Soft proximity weight: ~1 when close to target, ~0 when far
    proximity = 1.0 / (1.0 + alpha_proximity * dist_sq)

    # 1. Main progress: drive toward target center (dense quadratic penalty on distance)
    goal_proximity = -w_goal * dist_sq

    # 2. Soft landing velocity penalty: active only near the target
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Continuous contact reward: both legs touching near target,
    #    moderated by current speed quality (slower = higher reward)
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    speed_sq = vx**2 + vy**2
    speed_quality = 1.0 / (1.0 + beta_speed * speed_sq)
    contact_reward = w_contact * both_legs_contact * proximity * speed_quality

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
