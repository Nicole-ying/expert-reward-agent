# 设计理由
（正常模式：信号齐全，但主 shaping 信号 progress 被 landing_quality 的稀疏尖峰和惩罚掩盖，导致 agent 无法稳定靠近目标。仅增大 `w_progress` 从 5.0 到 20.0，使单位步长的前进奖励 (progress) 与其他组件（尤其是持续惩罚）可比，恢复 progress 作为主驱动信号的地位。不改变其他组件，保持着陆段的精细化约束。预期会提升 len 和 score，因为 agent 将被更平滑地引导到平台附近，之后再由 landing_quality 和速度惩罚负责最终着陆。）

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

    # 超参数 —— 仅 w_progress 由 5.0 → 20.0
    w_progress = 20.0        # 上调
    w_landing = 2.0
    w_land_vel = 10.0
    w_angle = 0.5
    w_angvel = 0.5
    engine_cost = 0.02

    # 距离计算
    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5

    # 1. 主学习信号：距离改进（potential‑based shaping）
    progress = w_progress * (dist - ndist)

    # 2. 着陆质量软信号（连续因子 + 几何平均，包含接触偏置）
    altitude_factor = max(0.0, 1.0 - abs(ny) / 0.2)
    align_factor    = max(0.0, 1.0 - abs(nx) / 0.2)
    vx_factor       = max(0.0, 1.0 - abs(nvx) / 0.3)
    vy_factor       = max(0.0, 1.0 - abs(nvy) / 0.5)
    angle_factor    = max(0.0, 1.0 - abs(nangle) / 0.2)
    contact_factor  = 0.1 + 0.9 * (nleft_contact + nright_contact) * 0.5

    product = (altitude_factor * align_factor * vx_factor *
               vy_factor * angle_factor * contact_factor)
    if product > 0.0:
        landing_quality = w_landing * (product ** (1.0 / 6.0))
    else:
        landing_quality = 0.0

    # 3. 着陆速度惩罚（仅在接触时）
    fcontact = float(nleft_contact * nright_contact)
    if fcontact > 0.5:
        vel_pen = -w_land_vel * (nvx**2 + nvy**2)
    else:
        vel_pen = 0.0

    # 4. 姿态稳定惩罚（全程）
    att_penalty = -w_angle * (nangle**2) - w_angvel * (nangvel**2)

    # 5. 引擎使用惩罚（节省燃料）
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
- **audit**: 所有观测维度已被使用，信号覆盖无缺口；landing_quality 稀疏尖峰主导奖励，progress 过于微弱，导致校准错误。
- **behavior**: agent 在接近平台附近获得巨大的稀疏奖励脉冲，但日常行为被持续惩罚压制，导致整体 epoch 长度缩短、得分停滞。
- **signal**: progress 的每步奖励远小于惩罚和 landing_quality 的尖峰，无法稳定塑造轨迹——主信号被淹没。
- **level**: Level 1
- **hypothesis**: 将 progress 系数提升 4 倍，使其每步贡献与持续惩罚相当或更高，恢复 progress 作为主导引力，让 agent 先学会顺畅接近平台，而着陆精细控制仍由现有组件在末端提供。
- **risk**: 若 agent 过度追求快速接近而忽视稳定性，后期可能需要轻微加强姿态/速度约束；但当前姿态惩罚和触地速度惩罚仍保留，风险可控。