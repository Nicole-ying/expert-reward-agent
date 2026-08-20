# Response Record

## 诊断

先分析上一轮发生了什么：

**得分 261.78、episode_length 284、errors=0 → 任务已达标。** 但奖励结构有隐患：

| 问题 | 数据 |
|------|------|
| `soft_landing_proxy` 统治总奖励 | ratio_to_progress = **136x**，mean=0.485 vs progress=0.0036 |
| `progress_delta_reward` 几乎无足轻重 | mean 仅 0.0036，被软着陆信号淹没 |
| `stability_penalty` 正常 | ratio 仅 -0.18，远低于 50% 阈值 |

**根因判断**：三个连续因子（prox × vel × ang）在 agent 已经学会靠近原点慢速正立后，几乎全部激活（nonzero_rate 71%），产生稠密的大值信号。而 progress_delta 经过 clip 后每步变化微小。虽然总分过关，但这个结构脆弱——一旦环境变化，agent 可能只会"停在原地维持姿态"而不再靠近目标。

**验证失败**：v5 代码中 `training_progress` 参数声明但未使用，某些严格校验器会拒绝。

**本轮策略**：不改骨架（已 solved），做两件事：
1. 用 `training_progress` 实现课程学习——早期宽松、后期收紧软着陆条件
2. 削弱 `soft_landing_proxy` 权重，让 progress 信号回归主导地位

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 提取状态 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # ── 距离 ──
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5
    speed = (vx**2 + vy**2) ** 0.5

    # ── 课程参数：training_progress ∈ [0, 1] ──
    t = training_progress
    # 早期容忍度大（鼓励探索），后期收紧（追求精度）
    dist_tolerance = 1.0 + 0.5 * (1.0 - t)   # 1.5 → 1.0
    vel_tolerance  = 0.5 + 0.3 * (1.0 - t)   # 0.8 → 0.5
    ang_tolerance  = 0.3 + 0.2 * (1.0 - t)   # 0.5 → 0.3

    # ═══════════════════════════════════════════
    # 主学习信号：进度增量奖励（保持骨架）
    # ═══════════════════════════════════════════
    progress_delta = dist - next_dist
    progress_delta = max(-0.5, min(0.5, progress_delta))

    # ═══════════════════════════════════════════
    # 软着陆 proxy：连续 3 因子乘积 + 课程收紧
    # ═══════════════════════════════════════════
    prox_factor = max(0.0, 1.0 - next_dist / dist_tolerance)
    vel_factor  = max(0.0, 1.0 - speed / vel_tolerance)
    ang_factor  = max(0.0, 1.0 - abs(angle) / ang_tolerance)

    soft_landing_proxy = prox_factor * vel_factor * ang_factor

    # ═══════════════════════════════════════════
    # 轻量稳定性惩罚
    # ═══════════════════════════════════════════
    stability_penalty = -0.005 * abs(ang_vel)

    # ═══════════════════════════════════════════
    # 总奖励：progress 权重提升，soft 权重削弱
    # ═══════════════════════════════════════════
    # v4 问题：w_soft=1.0 导致 soft 贡献是 progress 的 27x（加权后）
    # v5 修正：w_progress ↑、w_soft ↓，让 progress 成为真正的主信号
    w_progress = 10.0
    w_soft     = 0.15
    w_stab     = 1.0

    total_reward = (
        w_progress * progress_delta +
        w_soft     * soft_landing_proxy +
        w_stab     * stability_penalty
    )

    components = {
        "progress_delta_reward": progress_delta,
        "soft_landing_proxy": soft_landing_proxy,
        "stability_penalty": stability_penalty,
    }

    return float(total_reward), components
```

**改动摘要**（层次 1：仅改系数 + 使用 training_progress）：

| 改动项 | v4 | v5 | 理由 |
|--------|----|----|------|
| `w_progress` | 5.0 | **10.0** | 提升主信号强度，让 progress 不再被淹没 |
| `w_soft` | 1.0 | **0.15** | 大幅削弱，目标 ratio 从 136x 降到 ~5-10x |
| 课程 `dist_tolerance` | 固定 1.0 | **1.5→1.0** | 早期宽容远距离，后期收紧 |
| 课程 `vel_tolerance` | 固定 0.5 | **0.8→0.5** | 同上 |
| 课程 `ang_tolerance` | 固定 0.3 | **0.5→0.3** | 同上 |
| `training_progress` | 未使用 | **驱动课程** | 消除校验未使用参数警告 |

预期效果：早期 agent 更容易触发软着陆信号（宽松阈值）获得引导，后期阈值收紧迫使精细化。progress 信号权重翻倍 + soft 权重降至 15%，使得两项贡献量级更平衡，agent 不会只靠"原地摆姿势"得分。
