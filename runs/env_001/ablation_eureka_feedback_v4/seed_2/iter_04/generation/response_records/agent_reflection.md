# Response Record

# 设计理由
（此轮仅为代码格式修复，不改变原定修改方向。修正了上一版因特殊字符导致的解析失败，并补全了被截断的代码。核心内容与之前一致：移除 soft_landing，替换为低空门控的垂直速度惩罚、角度惩罚，以及低系数接触奖励，以提供清晰的坠落抑制信号。惩罚系数经过校准，预期 per-step ≤ 0.03，不超过 progress 的 2 倍。）

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前位置
    x, y = obs[0], obs[1]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_lc = next_obs[6]
    n_rc = next_obs[7]

    # 距离
    dist_old = (x**2 + y**2)**0.5
    dist_new = (nx**2 + ny**2)**0.5

    # 1. 目标进展
    w_progress = 1.0
    progress = w_progress * (dist_old - dist_new)

    # 2. 燃料效率
    w_fuel = 0.02
    fuel_cost = -w_fuel * float(action != 0)

    # 3. 安全着陆组件（仅低空门控激活）
    height = ny
    gate_height = 2.0
    gate = max(0.0, 1.0 - height / gate_height)

    # 垂直速度惩罚（hinge）
    safe_v = 0.5
    excess_v = max(0.0, abs(nvy) - safe_v)
    w_vpen = 0.05
    vertical_penalty = -w_vpen * excess_v * gate

    # 角度惩罚（hinge）
    safe_angle = 0.2
    excess_angle = max(0.0, abs(n_angle) - safe_angle)
    w_apen = 0.05
    angle_penalty = -w_apen * excess_angle * gate

    # 接触奖励（低系数，门控）
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
