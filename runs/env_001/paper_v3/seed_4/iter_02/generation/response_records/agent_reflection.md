# Response Record

`evidence`：全部20条评估episode均被截断（1000步），无一次成功或失败终止；proximity的episode_sum_mean高达792.55且active_rate=100%，说明智能体持续在目标附近徘徊获取状态奖励；soft_landing_bonus的active_rate=0%，双腿从未同时触地；velocity_penalty和attitude_penalty尺度极小（总和约-0.32），无法改变行为；外部score仅1.65，与内部奖励严重脱节。

`behavior_diagnosis`：智能体学会在目标平台正上方悬停，通过proximity（1/(1+dist)）每步获取约0.79的稳定奖励，累积1000步后内部奖励极高但从未尝试下降触地；双腿接触从未发生，说明智能体保持在平台面以上安全距离，完美利用了状态型奖励可无限累积的漏洞。

`signal_completeness`：主引导信号proximity是状态值而非改进量，存在state_to_improvement缺陷；着陆质量信号soft_landing_bonus以contact为硬性前提（乘法因子），在接触从未发生的情况下完全静默，形成sparse dead signal；缺失一个不依赖接触、能在下降过程中提供连续反馈的靠近地面信号；速度与姿态惩罚因系数过小形同虚设。

`selected_level`：Level 2 — soft_landing_bonus同时命中"sparse_to_dense"（contact硬条件导致active_rate=0%，缺少下降过程中的局部反馈）和"state_to_improvement的对称问题"（proximity可悬停累积但无信号鼓励继续下降到触地），需要将接触前提替换为高度连续信号，使智能体在下降过程中获得递增反馈。

`selected_intervention`：唯一修改soft_landing_bonus组件——将其从`contact_avg * factor_vel * factor_angle`（依赖接触的稀疏事件）改为`height_factor * factor_vel * factor_angle`（基于高度的连续下降质量信号），其中`height_factor = 1/(1+abs(y))`，同时将权重从1.0提升至3.0以使下降信号足以与悬停proximity竞争。

`falsifiable_hypothesis`：加入基于高度的连续下降奖励后，智能体在接近目标时将获得"下降比悬停更有利"的梯度信号——y减小→height_factor增大→descent_quality上升，从而打破纯悬停均衡；一旦智能体开始下降到y≈0附近，双腿接触传感器将首次被触发，为后续轮次引入真正的着陆信号创造条件。

`expected_next_round`：soft_landing_bonus组件（更名为descent_quality后）的active_rate应从0%显著上升至接近100%；episode_sum_mean应出现非零值且在下降阶段逐步增长；score应有实质性改善（当前仅1.65）；可能出现部分episode在平台高度附近徘徊但尚未完成硬着陆（如果接触仍需更低高度），这是可接受的中间状态。

`main_risk`：height_factor在y=0处最大（1.0）但双腿接触可能需要y略低于0，智能体可能学会在y=0处悬停而非继续下降到触发接触；若发生此情况，下一轮需要引入contact作为height_factor的补充乘数或添加微小增量奖励。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from the next observation
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

    # Descent quality: continuous reward for being near ground level
    # with low speed and small attitude deviations
    # Replaces the previous contact-gated soft_landing_bonus that never fired
    height_factor = 1.0 / (1.0 + abs(y))    # peaks at y=0 (platform surface)
    speed_norm = (vx**2 + vy**2) ** 0.5
    factor_vel = 1.0 / (1.0 + speed_norm)
    factor_angle = 1.0 / (1.0 + abs(angle) + abs(angular_vel))
    descent_quality = height_factor * factor_vel * factor_angle
    w_descent = 3.0
    comp_descent = w_descent * descent_quality

    # Quadratic penalties for high velocity and attitude deviations
    w_vel_pen = 0.01
    vel_pen = -w_vel_pen * (vx**2 + vy**2)

    w_att_pen = 0.01
    att_pen = -w_att_pen * (angle**2 + angular_vel**2)

    total = comp_prox + comp_descent + vel_pen + att_pen

    components = {
        'proximity': comp_prox,
        'descent_quality': comp_descent,
        'velocity_penalty': vel_pen,
        'attitude_penalty': att_pen,
    }
    return float(total), components
```
