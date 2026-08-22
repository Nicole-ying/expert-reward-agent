# 设计理由
本轮修改 **landing_quality 组件**，从原先的二值稀疏 proxy（仅双腿同时接触且多条件满足时才激活，active_rate 仅 0.2%）改造为**连续 bounded factor + 几何平均**的 dense 引导信号。  
- **为什么改**：原 landing_quality 几乎从未触发，对 agent 着陆行为无任何优化信号；agent 只会无目标地靠近平台（仅靠 progress），无法学习精确着陆。  
- **数学形式**：用六个因子的几何平均构建连续质量指标：  
  - `altitude_factor`（y 接近 0）、`align_factor`（x 接近 0）、`vx_factor`、`vy_factor`、`angle_factor`（各状态在其阈值内线性衰减到 1）  
  - `contact_factor = 0.1 + 0.9 × (left_contact + right_contact) × 0.5`，即使未接触也保留底值 0.1，避免早停塌缩，同时双腿接触时接近 1.0 产生显著增益。  
- **系数校准**：保持 w_landing=2.0，因子几何平均后预期最大约 1.0，单步奖励峰值 ≤2.0，着陆阶段整体贡献预计不超过 progress 的 2 倍，符合单组件上限约束。  

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

    # 超参数
    w_progress = 5.0
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
- **audit**: 所有观测已使用，核心问题是 landing_quality 信号缺失（active_rate 0.2%）而非信号校准；僵尸组件必须重建。  
- **behavior**:  agent 能够下降靠近平台，但缺乏着陆引导，极少双足接触，姿态与引擎惩罚偏重进一步压制尝试。  
- **signal**: 缺少目标平台附近的密集着陆质量信号，原接触条件过于严格，导致 agent 无法感知“怎样才算好着陆”。  
- **level**: Level 2  
- **hypothesis**: 连续化的 landing_quality（含接触偏置）将在接近平台时提供稳定梯度，引导 agent 在最终阶段减速对齐、伸出双腿，从而提升着陆成功率，驱动总分向目标靠近。  
- **risk**: 若 w_landing 过大，agent 可能在平台上方徘徊获取奖励而不完全着陆；后续需根据 active_rate 与长度变化微调系数或加入接触强制激励。