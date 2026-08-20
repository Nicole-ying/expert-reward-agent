## 诊断

### 1. Agent 发生了什么？
**Crash** — episode 长度仅 73.6 步（< 150），10/10 全部提前终止，得分 -108.94。Agent 无法存活到自然结束。

### 2. 哪个组件是主要原因？
**`stability_penalty`** — 其 `ratio_to_progress = 9.18`，惩罚均值 0.147 是主信号 progress_delta (0.016) 的 **9 倍**。根据原则 1，惩罚超过主信号 50% 时 agent 会选择"不动"来避免惩罚。Agent 被速度/角度/角速度惩罚压垮，无法有效探索。

`soft_landing_bonus` 触发率仅 0.2%（nonzero_rate=0.002），这是结果而非原因——因为 penalty 压制了探索，agent 根本到不了目标附近。

### 3. 上一轮做了什么？
这是第一轮（iter 1），无历史。骨架是 `progress_delta + soft_landing_bonus + stability_penalty`，需要先解决 penalty 压倒性问题。

### 修改方案：层次 1 — 削弱 stability_penalty 系数
当前权重 w_vel=0.1, w_angle=0.5, w_angvel=0.1 使 penalty 均值达 0.147。目标是将 penalty 均值压到 progress 的 30-50% 以下（约 0.005-0.008），需要约 **20-25x 削弱**。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress delta（到目标垫中心的距离减少量）
    x_prev, y_prev = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    d_prev = (x_prev**2 + y_prev**2) ** 0.5
    d_next = (x_next**2 + y_next**2) ** 0.5
    progress_delta = d_prev - d_next

    # 稳定约束：大幅削弱权重，避免压制 progress 信号
    # 原权重 (0.1, 0.5, 0.1) 使 penalty 均值 0.147 = progress 的 9x
    # 削弱 ~25x → 目标 penalty 均值 ~0.006 = progress 的 ~40%
    x_vel, y_vel = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    w_vel = 0.004      # 原 0.1 → 削弱 25x
    w_angle = 0.02     # 原 0.5 → 削弱 25x
    w_angvel = 0.004   # 原 0.1 → 削弱 25x
    stability_penalty = (
        w_vel * (abs(x_vel) + abs(y_vel)) +
        w_angle * abs(body_angle) +
        w_angvel * abs(ang_vel)
    )

    # 任务完成软代理：接近中心 + 低速 + 水平 + 双脚着地
    # 当前触发率极低（0.2%）是因 penalty 压制导致无法靠近目标，先不调整
    near = (d_next < 0.1)
    slow = (abs(x_vel) < 0.1 and abs(y_vel) < 0.1)
    level = (abs(body_angle) < 0.1)
    both_feet = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    soft_landing_bonus_raw = 1.0 if (near and slow and level and both_feet) else 0.0
    soft_landing_bonus = 0.5 * soft_landing_bonus_raw

    total_reward = progress_delta - stability_penalty + soft_landing_bonus

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```