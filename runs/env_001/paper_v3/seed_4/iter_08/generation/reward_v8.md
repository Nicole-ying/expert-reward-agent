`evidence`：当前策略得分-34.29远差于best，episode几乎全部truncated（19/20），说明在时间上限附近徘徊。descent_quality贡献96.68（64.2% signed share），proximity delta仅36.75（24.4%），penalties近乎为零（合计-0.27），contact_quality几乎不触发（0% active_rate）。外部得分与训练奖励严重背离：奖励总和约133.6但score为-34.29，说明agent正在通过在高空保持低速度低姿态来大量收割descent_quality，同时缓慢靠近目标获取少量proximity delta，但始终不真正着陆。上一轮将proximity从状态值改为差分，同时引入contact_quality和contact_gate，结果失败。

`behavior_diagnosis`：agent学会在离目标一定距离处悬停（极低速度v²≈0.021、近零姿态angle²+angvel²≈0.006），通过height_factor≈1持续收割descent_quality奖励，同时缓慢漂移获取净正的proximity delta，但始终不完成着陆，直到1000步被截断。这是典型的dense proxy farming。

`signal_completeness`：proximity（状态值）提供到达目标的持续激励，descent_quality提供着陆姿态引导，penalties提供速度和姿态约束。核心缺失：descent_quality是全局稠密信号，在远离目标的任何位置只要贴近地面就能被收割，与任务完成（在目标平台着陆）脱节。contact相关激励几乎不可达（active_rate=0%）。

`selected_level`：Level 2 — `dense_to_task_event` / `proxy_to_completion_alignment`。证据明确：dense proxy（descent_quality）被大量收割但外部任务失败。单纯降系数（Level 1）无法消除"悬停即得分"的根本漏洞，因为只要descent_quality存在正值，agent就偏好无限悬停而非终止。需要localize这个proxy，使其只在靠近目标时生效。

`selected_intervention`：以best代码为基础，对descent_quality施加proximity_to_target门控。将其从全局稠密信号变为仅在靠近目标平台时有效的局部着陆引导信号。不引入contact_quality和contact_gate（上一轮失败变更），保持best的简洁结构。

`falsifiable_hypothesis`：门控后，远离目标时descent_quality≈0，agent无法再通过远距离悬停收割该奖励；proximity（状态值，w=1.0）将成为到达目标的主要驱动力；靠近目标后descent_quality逐渐解锁，提供着陆姿态引导。agent应优先到达目标区域，再在目标附近降低高度、完成着陆。

`expected_next_round`：descent_quality的episode_sum_mean和magnitude_share应显著下降（尤其来自远距离的贡献被截断）；proximity的signed_share应上升成为主导；episode平均长度可能缩短（若成功着陆增加）或保持（若仍需探索着陆）；外部score应从-34向正方向改善；terminated比例应上升。

`main_risk`：门控后descent_quality在远距离时接近零，早期探索阶段agent可依赖的稠密信号减少，可能导致学习变慢或陷入局部最优。若proximity（状态值）自身在远距离时梯度不足，早期接近行为可能退化。

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

    # Proximity to target (assumed at origin) — state-based, unchanged from best
    dist = (x**2 + y**2) ** 0.5
    proximity = 1.0 / (1.0 + dist)          # bounded [0,1]
    w_proximity = 1.0
    comp_prox = w_proximity * proximity

    # Descent quality — LOCALIZED by proximity to target
    # Gate prevents proxy farming: good descent posture only rewarded near the target
    prox_to_target = 1.0 / (1.0 + dist)     # same as proximity, reuse
    height_factor = 1.0 / (1.0 + abs(y))    # peaks at y=0 (platform surface)
    speed_norm = (vx**2 + vy**2) ** 0.5
    factor_vel = 1.0 / (1.0 + speed_norm)
    factor_angle = 1.0 / (1.0 + abs(angle) + abs(angular_vel))
    descent_quality = prox_to_target * height_factor * factor_vel * factor_angle
    w_descent = 3.0
    comp_descent = w_descent * descent_quality

    # Quadratic penalties for high velocity and attitude deviations (unchanged from best)
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