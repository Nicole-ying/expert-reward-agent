# 设计理由
当前 reward 已使 agent 稳定完成着陆，score 大幅超越目标。但组件中存在轻微冗余：`landing_velocity_penalty` 与 `landing_quality` 内的速度因子（`vx_factor`、`vy_factor`）功能重叠——两者都抑制着陆时的水平/垂直速度。`landing_velocity_penalty` 的 episode_sum_mean 仅 -0.306，贡献极低，保留它无益且可能干扰 `landing_quality` 的梯度传导。将其系数置零（等效移除），使着陆阶段的减速动机完全由 `landing_quality` 的连续因子承担，简化信号结构，同时保持原有高性能不变。

修改方式：将 `w_land_vel` 从 10.0 调整为 0.0，保留组件键名以维持外部接口兼容，但实际贡献恒为零。

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
    w_landing = 0.3
    w_land_vel = 0.0            # 冗余速度惩罚已停用，动机完全由 landing_quality 承接
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

    # 3. 着陆速度惩罚（已停用）
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
- **audit**: 所有观测维度均被使用，无信号缺口；agent 终止于成功着陆，当前已超目标分数，评分稳定。
- **behavior**: agent 能以 284 步左右快速着陆，主引擎和方向引擎配合良好， landing 过程平稳。
- **signal**: `landing_velocity_penalty` 与 `landing_quality` 速度因子功能重叠，前者贡献近乎为零，构成冗余。
- **level**: Level 2（移除套娃型冗余组件）
- **hypothesis**: 精简后，速度抑制完全由 `landing_quality` 的连续因子承担，梯度传导更干净，性能至少持平，可能微升。
- **risk**: 着陆速度敏感性略微降低，但 `landing_quality` 的几何平均仍提供强力减速动机，无实质退化风险。