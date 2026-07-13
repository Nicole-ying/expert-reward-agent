**1. evidence**：当前得分71.5，7/20成功终止、13/20被截断；contact_bonus以50.0单次触发、active_rate 6.2%意味着每episode平均触发约57次，magnitude_share高达51.7%，远超过descent_quality的35.6%；velocity_penalty和attitude_penalty的magnitude_share均为0.0%，系数0.01过小完全失效。

**2. behavior_diagnosis**：agent在平台附近反复弹跳，利用"非双脚→双脚"的状态转换反复领取contact_bonus（每次50，约57次/episode），而非稳定着陆；这解释了13/20的truncation（长时间存活但不终止）和得分方差极大（-7.4到193.9）。

**3. signal_completeness**：proximity和descent_quality提供了基本的引导和软着陆信号，职责基本完备；但contact_bonus作为过渡事件奖励的数学形态被反复触发，构成exploit；velocity/attitude惩罚因系数过小基本不存在。

**4. selected_level**：Level 2 — contact_bonus的数学形态（稀疏过渡事件）在当前行为下被反复利用，不是单纯尺度问题，需要结构变换。

**5. selected_intervention**：将contact_bonus从基于状态转移的稀疏事件（`not prev_both and curr_both → +50`）改为持续接触质量奖励（双脚着地时`factor_vel * factor_angle`，系数5.0），消除重复触发漏洞。

**6. falsifiable_hypothesis**：新的持续接触质量奖励只在双脚着地且低速度、小角度时给出正向反馈，弹跳离开地面时立即归零，应消除"弹跳刷分"行为，促使agent在触地后保持稳定直至终止。

**7. expected_next_round**：contact组件的active_rate应显著下降（仅在实际双脚着地期间激活），episode_sum_mean不再膨胀，成功终止率上升，truncation率下降，平均score提高。

**8. main_risk**：contact_quality与descent_quality在双脚着地时高度重叠（均依赖factor_vel和factor_angle），可能导致近地悬停而不实际触地的proxy徘徊；若下一轮验证出现此问题，需差异化两者的作用阶段。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from observations
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Proximity to target (assumed at origin)
    dist = (x**2 + y**2) ** 0.5
    proximity = 1.0 / (1.0 + dist)          # bounded [0,1]
    w_proximity = 1.0
    comp_prox = w_proximity * proximity

    # Shared stability factors reused across components
    speed_norm = (vx**2 + vy**2) ** 0.5
    factor_vel = 1.0 / (1.0 + speed_norm)       # bounded [0,1], 1 when still
    factor_angle = 1.0 / (1.0 + abs(angle) + abs(angular_vel))  # bounded [0,1]

    # Descent quality: reward being near ground, slow, and upright
    height_factor = 1.0 / (1.0 + abs(y))        # peaks at y=0
    descent_quality = height_factor * factor_vel * factor_angle
    w_descent = 3.0
    comp_descent = w_descent * descent_quality

    # Sustained contact quality: reward stable, settled contact with both feet
    # Replaces the transition-based contact_bonus that was exploited via bouncing
    both_contact = (left_contact + right_contact) >= 1.5
    if both_contact:
        contact_quality = factor_vel * factor_angle   # bounded [0,1], high when stable
        w_contact = 5.0
        comp_contact = w_contact * contact_quality
    else:
        comp_contact = 0.0

    # Quadratic penalties for high velocity and attitude deviations
    w_vel_pen = 0.01
    vel_pen = -w_vel_pen * (vx**2 + vy**2)

    w_att_pen = 0.01
    att_pen = -w_att_pen * (angle**2 + angular_vel**2)

    total = comp_prox + comp_descent + comp_contact + vel_pen + att_pen

    components = {
        'proximity': comp_prox,
        'descent_quality': comp_descent,
        'contact_quality': comp_contact,
        'velocity_penalty': vel_pen,
        'attitude_penalty': att_pen,
    }
    return float(total), components
```