# 设计理由
我将垂直速度惩罚从全局生效改为低空门控，只在 `ny < 2.0` 的高度范围内激活。原始惩罚因在整个下降过程中连续扣分，与 progress 形成等量抵消，导致 agent 不敢快速接近目标。通过引入 `gate_v = max(0.0, 1.0 - ny / 2.0)`，高空的垂直速度不再被惩罚，agent 可以自由加速下降以积累更多 progress，进入低空后门控渐强，仍能引导末端减速，因此预期能改善软着陆行为并提升总得分。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_lc = next_obs[6]
    n_rc = next_obs[7]

    dist_old = (x**2 + y**2)**0.5
    dist_new = (nx**2 + ny**2)**0.5

    w_progress = 1.0
    progress = w_progress * (dist_old - dist_new)

    w_fuel = 0.02
    fuel_cost = -w_fuel * float(action != 0)

    # 垂直速度惩罚，仅在接近地面时生效
    safe_v = 0.5
    excess_v = max(0.0, abs(nvy) - safe_v)
    w_vpen = 0.05
    gate_v = max(0.0, 1.0 - ny / 2.0)
    vertical_penalty = -w_vpen * excess_v / (1.0 + excess_v) * gate_v

    gate = max(0.0, 1.0 - ny / 2.0)
    safe_angle = 0.2
    excess_angle = max(0.0, abs(n_angle) - safe_angle)
    w_apen = 0.05
    angle_penalty = -w_apen * excess_angle * gate

    w_contact = 0.05
    contact_bonus = w_contact * (n_lc + n_rc) * gate

    total_reward = progress + fuel_cost + vertical_penalty + angle_penalty + contact_bonus
    components = {
        "progress": progress,
        "fuel_cost": fuel_cost,
        "vertical_penalty": vertical_penalty,
        "angle_penalty": angle_penalty,
        "contact_bonus": contact_bonus
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated=100%，episode长度68，progress=1.11，vertical_penalty=-0.927，二者量级相抵。
- **behavior**: agent 在整个下降过程中被垂直速度惩罚持续扣分，导致 progress 积累缓慢，在尚未显著靠近目标时就因碰撞终止。
- **signal**: 全局垂直速度惩罚与 progress 形成对抗，抑制了下降效率，缺少仅在地面附近强制减速的引导。
- **level**: Level 2（结构变换：全局惩罚→低空门控）
- **hypothesis**: 移除高空垂直速度惩罚后，agent 将更积极地下降以获取更大的 progress，进入低空后自动减速，从而提升总得分和着陆成功率。
- **risk**: 若 agent 在高空加速过度以至于到达低空时速度远超安全阈值，可能来不及减速并 crash；但 gate 的渐进衰减和 hinge 上限可提供裕度，风险可控。