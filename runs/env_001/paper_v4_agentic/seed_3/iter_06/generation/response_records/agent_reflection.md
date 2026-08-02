# Response Record

# 设计理由
当前奖励函数已经将得分推到 241.7，远超目标 200，且大部分 episode 以成功着陆终止（terminated 18/20，len 443）。信号覆盖审计显示所有观测维度均被使用，无僵尸组件。唯一的小瑕疵是 `landing_quality` 的 episode_sum_mean 高达 50.28，占 signed share 56.6%，而 `progress` 只有 31.1%。这种过强的着陆奖励可能让 agent 在最终接近平台时过度精细调姿，拉长 episode，并可能压制 `progress` 的早期引导。本次做 **Level 1 尺度修复**：将 `w_landing` 从 0.5 降至 0.3，使着陆奖励仍为正但份额下降，让 agent 更平衡地关注快速接近与柔和着陆。其他组件不变。

系数校准：`landing_quality` per‑episode 从 ~50 预计降至 ~30，与 `progress` (~28) 更均衡。`w_landing` 0.3 依然能提供足够梯度，不会导致 crash 率上升。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆分
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft_contact = next_obs[6]
    nright_contact = next_obs[7]

    w_progress = 20.0
    w_landing = 0.3            # 从 0.5 降至 0.3，平衡着陆奖励份额
    w_land_vel = 10.0
    w_angle = 0.5
    w_angvel = 0.5
    engine_cost = 0.02

    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5

    # 1. 距离改进
    progress = w_progress * (dist - ndist)

    # 2. 着陆质量软信号 —— 仅在双腿同时接触时激活
    if nleft_contact > 0.5 and nright_contact > 0.5:
        altitude_factor = max(0.0, 1.0 - abs(ny) / 0.5)
        align_factor    = max(0.0, 1.0 - abs(nx) / 0.5)
        vx_factor       = max(0.0, 1.0 - abs(nvx) / 0.3)
        vy_factor       = max(0.0, 1.0 - abs(nvy) / 0.5)
        angle_factor    = max(0.0, 1.0 - abs(nangle) / 0.2)
        product = (altitude_factor * align_factor * vx_factor *
                   vy_factor * angle_factor)
        if product > 0.0:
            landing_quality = w_landing * (product ** (1.0 / 5.0))
        else:
            landing_quality = 0.0
    else:
        landing_quality = 0.0

    # 3. 着陆速度惩罚（仅在双腿接触时）
    fcontact = float(nleft_contact * nright_contact)
    if fcontact > 0.5:
        vel_pen = -w_land_vel * (nvx**2 + nvy**2)
    else:
        vel_pen = 0.0

    # 4. 姿态稳定惩罚（全程）
    att_penalty = -w_angle * (nangle**2) - w_angvel * (nangvel**2)

    # 5. 引擎使用惩罚
    eng_pen = -engine_cost if action != 0 else 0.0

    total_reward = progress + landing_quality + vel_pen + att_penalty + eng_pen
    components = {
        "progress": progress,
        "landing_quality": landing_quality,
        "landing_velocity_penalty": vel_pen,
        "attitude_penalty": att_penalty,
        "engine_cost": eng_pen
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测均被使用，无僵尸组件；termination 多为成功着陆，奖励函数工作良好，仅需平衡份额。
- **behavior**: agent 已学会快速着陆（len≈443）并稳定获取高分，偶尔 truncation，无早逝或剥削迹象。
- **signal**: landing_quality 占据 56.6% 的正信号，过强可能挤占 progress 的早期引导。  
- **level**: Level 1  
- **hypothesis**: 降低 w_landing 至 0.3 使着陆奖励份额与 progress 更均衡，agent 会更快接近平台而不牺牲着陆质量。  
- **risk**: 若着陆奖励过弱，agent 可能忽略减速导致 crash；但 0.3 仍提供足够的正梯度，风险较低。
