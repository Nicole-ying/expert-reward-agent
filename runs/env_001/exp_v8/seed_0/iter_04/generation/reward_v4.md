## 诊断

**Agent 状态**：Score 169.76，len 922。Agent 不再 crash，存活良好，从 iter 2 的 -108 大幅跃升。

**组件分析**：

| 组件 | ratio | nonzero_rate | 判断 |
|------|-------|-------------|------|
| `progress_reward` | 1.0 (基准) | 99.8% | ✅ 正常工作 |
| `soft_landing_proxy` | **6.9** | 41.5% | 🔴 **严重主导** — 是 progress 的 7 倍，agent 主要在追这个 proxy |
| `stability_penalty` | ~0 | 0.35% | 🟡 几乎死信号，但非本轮重点 |

**根因**：`soft_landing_proxy` 的连续乘积形式天然产出 [0,1] 值域，未经缩放直接加入 total，使其 magnitude (~0.29) 远超 progress_reward (~0.043)。这会导致 agent 优化 proxy 条件而非真正的 progress。

**本轮修改（只改一处，遵循原则 3）**：给 `soft_landing_proxy` 加缩放系数 0.15，将 ratio 从 6.9 压到 ~1.0，使其回归辅助信号角色。不碰 stability_penalty（下轮再处理）。

关于 validation 失败：上一版 v4 提交的具体代码未知，但我基于已验证通过的 v3 骨架，仅做系数调整，确保语法和索引安全。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标（scale=8，符合 skeleton 推荐 5~20）
    progress_reward = (d_curr - d_next) * 8.0

    # 2. 轻量稳定约束
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))
    angular_v = abs(float(next_obs[5]))

    raw_stability_penalty = (
        -0.005 * speed
        - 0.01 * angle
        - 0.005 * angular_v
    )

    # stage weighting: 早期忽略 stability，后期逐渐加入
    # t=0 → late_weight=0（无惩罚）；t=1 → late_weight=1（全惩罚）
    t = training_progress
    late_weight = max(0.0, min(1.0, 2.0 * t - 0.5))
    stability_penalty = raw_stability_penalty * late_weight

    # 3. 软着陆近似信号：连续乘积因子提供梯度
    # 每个因子 bounded 在 [0,1]
    proximity_factor = max(0.0, 1.0 - d_next / 0.3)       # 距离 < 0.3 时 >0
    speed_factor = max(0.0, 1.0 - speed / 0.5)            # 速度 < 0.5 时 >0
    angle_factor = max(0.0, 1.0 - angle / 0.3)            # 角度 < 0.3rad 时 >0
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )  # 两腿着地程度乘积

    soft_landing_proxy = (
        proximity_factor * speed_factor * angle_factor * contact_factor
    )

    # ★ 本轮唯一改动：缩放 soft_landing_proxy，将其 ratio 从 6.9 压至 ~1.0
    # 系数 0.15 = target_ratio(1.0) / current_ratio(6.9)
    scaled_landing_proxy = soft_landing_proxy * 0.15

    # 总奖励
    total_reward = progress_reward + stability_penalty + scaled_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": scaled_landing_proxy,
        "total_reward": total_reward,
    }
    return float(total_reward), components
```