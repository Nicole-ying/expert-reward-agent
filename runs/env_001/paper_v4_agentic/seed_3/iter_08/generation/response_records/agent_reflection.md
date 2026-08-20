# Response Record

# 设计理由
上一轮（iter 7）停用了 `landing_velocity_penalty`，期望由 `landing_quality` 内部的速度因子完全接管减速责任。实际结果：score 从 266.03 下跌至 235.52，且 `landing_quality` 的 episode_sum_mean 从 0.119 升至 0.170（有改善），但总分反而下降，说明部分 episode 在着陆瞬间速度仍过大，导致整体得分受损，也可能触发 crash‑like termination 拉低外部评估分。因此需要重新引入着陆速度约束，但采用更精准的干预方式。

本轮改动：**将 `landing_velocity_penalty` 从停用（`w_land_vel=0`、二次全时惩罚）改造为 hinge 式**——只在双腿同时接触平台时激活，仅对超出安全阈值（水平 0.2 m/s、垂直 0.3 m/s）的过量速度施加线性惩罚。系数设为 0.2，使每回合此惩罚的总量级不超过 `progress` 总奖励的 0.5 倍。这样既给予 agent 在接近阶段足够的机动自由，又在着陆立刻提供明确的减速梯度，避免因速度过大导致 landing 质量掉到 0 之外还产生额外负奖励。其余组件系数、逻辑保持不变。

## 系数校准
- `w_land_vel = 0.2`，per‑step 惩罚 ~0.05–0.1（仅最后 1–2 步激活），远低于 progress 的 per‑step ≈0.1。
- hinge 阈值 `vx=0.2`、`vy=0.3` 对应着陆安全的 70–80% 容忍边界，保证在“不理想但可接受”的条件下惩罚为零。

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
    w_land_vel = 0.2            # 重新激活，改为 hinge 式
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

    # 3. 着陆速度惩罚 (hinge) —— 仅双腿接触时激活，惩罚超出安全阈值的部分
    if nleft_contact > 0.5 and nright_contact > 0.5:
        excess_vx = max(0.0, abs(nvx) - 0.2)
        excess_vy = max(0.0, abs(nvy) - 0.3)
        vel_pen = -w_land_vel * (excess_vx + excess_vy)
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
- **audit**: 信号齐全；删除速度惩罚后着陆阶段缺乏独立减速梯度，导致得分下降。
- **behavior**: agent 能在 260 步左右完成着陆，但部分 episode 着陆时速度偏大，拉低 total reward。
- **signal**: 缺少着陆接触瞬间的过量速度惩罚（原组件已停用）。
- **level**: Level 2（结构变换：停用的二次惩罚 → hinge 惩罚，仅在接触时激活）
- **hypothesis**: hinge 式速度惩罚将只在着陆瞬间对超过安全阈值的速度施加负奖励，提供明确减速压力，同时不干扰早期探索；与 `landing_quality` 内部的因子形成互补，预期恢复并超越 iter 6 的性能。
- **risk**: 若阈值或系数过严，可能使 agent 过度减速延长 episode，但当前 0.2/0.3 阈值较宽松，且惩罚仅在最后一步生效，风险可控。
