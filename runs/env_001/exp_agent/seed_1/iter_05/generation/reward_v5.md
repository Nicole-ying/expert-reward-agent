## 诊断分析

### 1. Agent 发生了什么？
- 得分 235.96，略低于历史最佳 251.94（-6.4%），仍在 solved 区间（>200）
- Episode 长度从 363 → 479（+32%），agent 存活更久但效率下降
- 无 crash、无 error
- **模式**：agent 似乎在目标附近徘徊更久才着陆 — 典型的"部分信用 exploit"

### 2. 哪个组件是主因？
- `bounded_proximity` 不变（0.559 vs 0.560），不是问题
- `stability_penalty` 不变（0.003），不是问题
- **`soft_landing_bonus`**：mean 从 0.178 → 0.156，nonzero_rate 高达 51.2%

核心问题：上轮把二值（阈值 0.1）→ 连续乘积（阈值 0.2），阈值翻倍导致"软着陆奖励区域"扩大了一倍。Agent 在 d<0.2 的区域内就能获得部分信用，学会了在着陆点附近悬浮而非果断着陆。更长的 episode 直接印证了这一点。

### 3. 我之前改了什么？
- 上轮（iter 4）：soft_landing 二值 → 连续乘积，阈值从 0.1 放宽到 0.2
- 效果：得分微降，episode 变长，nonzero_rate 升高 → 阈值太宽

**修复方向**：阈值收紧回 0.1（匹配原始二值逻辑的触发条件），保留连续形式的梯度优势，但只在真正接近着陆的狭窄区域内生效。这既给梯度，又不给悬浮者发"参与奖"。

这是单一改动（只改阈值，不改权重），下一轮可清晰归因。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 主学习信号：bounded proximity（不变） ===
    x_next, y_next = next_obs[0], next_obs[1]
    d_next = (x_next ** 2 + y_next ** 2) ** 0.5
    k_proximity = 5.0
    bounded_proximity = 1.0 / (1.0 + k_proximity * d_next)

    # === 稳定约束（不变） ===
    x_vel, y_vel = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    w_vel = 0.004
    w_angle = 0.02
    w_angvel = 0.004
    stability_penalty = (
        w_vel * (abs(x_vel) + abs(y_vel)) +
        w_angle * abs(body_angle) +
        w_angvel * abs(ang_vel)
    )

    # === 软着陆奖励：连续乘积，阈值收紧至 0.1（匹配原二值条件） ===
    # 上一轮阈值 0.2 太宽，nonzero_rate 51% 说明 agent 在目标附近悬浮吃部分信用
    # 收紧到 0.1 后，只有真正接近着陆条件时才有非零信号，保留连续梯度优势
    near_factor = max(0.0, 1.0 - d_next / 0.1)
    speed = abs(x_vel) + abs(y_vel)
    slow_factor = max(0.0, 1.0 - speed / 0.1)
    level_factor = max(0.0, 1.0 - abs(body_angle) / 0.1)
    feet_factor = 0.5 * (next_obs[6] + next_obs[7])

    soft_landing_raw = near_factor * slow_factor * level_factor * feet_factor
    soft_landing_bonus = 0.5 * soft_landing_raw  # 权重不变

    total_reward = bounded_proximity - stability_penalty + soft_landing_bonus

    components = {
        "bounded_proximity": bounded_proximity,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
    }

    return float(total_reward), components
```