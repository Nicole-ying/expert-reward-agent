# Response Record

# 设计理由
原 vertical_penalty 使用了门控 `gate = max(0, 1 - height/2)`，导致高度 >2 时惩罚完全为零，agent 在高空无法学习控制垂直速度，直到逼近地面速度过大才触发惩罚且来不及纠正。本轮移除门控，将垂直速度限制变为全局生效；同时为防止极端速度（如坠落）产生无界奖励塌缩，将线性 hinge 改为有界形式 `excess_v / (1 + excess_v)`，系数设为 0.05，使典型步长惩罚 ≈ -0.017（与 progress 步均 0.016 相当），即使极端速度惩罚也不超过 -0.05，不支配总奖励。

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

    # 3. 垂直速度惩罚（全局有界，无门控）
    safe_v = 0.5
    excess_v = max(0.0, abs(nvy) - safe_v)
    w_vpen = 0.05
    vertical_penalty = -w_vpen * excess_v / (1.0 + excess_v)

    # 4. 角度惩罚（保留低空门控）
    height = ny
    gate = max(0.0, 1.0 - height / 2.0)
    safe_angle = 0.2
    excess_angle = max(0.0, abs(n_angle) - safe_angle)
    w_apen = 0.05
    angle_penalty = -w_apen * excess_angle * gate

    # 5. 接触奖励（保留低空门控）
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
- **evidence**: iter4 垂直速度惩罚均值 -1.22（步均 -0.018）主导负分，且 terminated 全为失败；门控导致高区无梯度。
- **behavior**: agent 在低空遭遇强烈速度惩罚，因高区缺乏减速信号，无法及时调整，最终 crash。
- **signal**: 垂直速度惩罚的早期信号缺失。
- **level**: Level 2（局部门控→全局有界惩罚，防止极端值支配）。
- **hypothesis**: 全程可微的垂直速度惩罚会让 agent 从高处开始学习控制下降速度，提高安全着陆概率。
- **risk**: 若系数偏高，可能使 agent 过度减速、徘徊不降，延长 episode；有界形式能避免极端速度时的奖励塌缩。
