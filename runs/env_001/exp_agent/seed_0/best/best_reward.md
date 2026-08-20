## 诊断

### 1. 这个 agent 发生了什么？

**明确失败。** Score=-1771（比历史最佳 -107 差 16 倍），60% 的 episode 在 150 步内 crash。

对比历史：
- iter 1（progress_delta 骨架）：-107，len=74，**可用**
- iter 3（potential_shaping 骨架）：-1771，len=213，**灾难**

根因不是参数问题，是**骨架选错了**。

### 2. 哪个组件是主要原因？

| 证据 | 值 | 判断 |
|---|---|---|
| `soft_landing_proxy` nonzero_rate | **0.15%** | 5 因子乘积（含 contact）几乎永远为 0，形同虚设 |
| `potential_shaping` mean | 0.130 | 唯一有效信号，但势能塑形在此环境引导力弱于 progress_delta |
| `stability` (iter 1) ratio_to_progress | **~9x** | 惩罚压倒主信号，agent 学会"不动" |

**核心问题：骨架从 progress_delta 切换到 potential_shaping 是方向性错误。**

### 3. 我之前改了什么？

iter 3 把整个骨架从 `progress_delta_reward` 换成了 `potential_shaping`，同时把 soft_landing 从二值改成 5 因子连续乘积。两个大改动叠加，完全破坏了 iter 1 还能工作的信号结构。

---

## 修改方案

**回归 progress_delta_reward 骨架**（唯一达到过 -107 的方向），在此基础上做三处精准修复：

1. **soft_landing 从二值 → 连续 3 因子**（去掉 contact 和 ang_vel，因为 contact 在飞行中永远为 0）
2. **stability 惩罚大幅削弱**（iter 1 中 penalty 是 progress 的 9 倍，远超 10% 安全线）
3. **progress_delta 加 mild clip** 防目标附近震荡

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

    # ── 速率标量 ──
    speed = (vx**2 + vy**2) ** 0.5

    # ═══════════════════════════════════════════
    # 主学习信号：进度增量奖励
    # ═══════════════════════════════════════════
    # 每一步向原点靠近就奖励，远离就惩罚
    progress_delta = dist - next_dist
    # mild clip 防目标附近震荡（skeleton 文档推荐）
    progress_delta = max(-0.5, min(0.5, progress_delta))

    # ═══════════════════════════════════════════
    # 软着陆 proxy：连续 3 因子乘积（无 contact）
    # ═══════════════════════════════════════════
    # 去掉 contact_factor —— 飞行中接触永远为 0，会让整个乘积归零
    # 去掉 ang_vel_factor —— 减少因子数，提升 nonzero_rate
    # 三个因子：靠近原点 + 低速 + 姿态正
    prox_factor = max(0.0, 1.0 - next_dist / 1.0)       # 距离 < 1.0 时有梯度
    vel_factor  = max(0.0, 1.0 - speed / 0.5)            # 速率 < 0.5 时有梯度
    ang_factor  = max(0.0, 1.0 - abs(angle) / 0.3)       # 倾角 < 0.3 时有梯度

    soft_landing_proxy = prox_factor * vel_factor * ang_factor

    # ═══════════════════════════════════════════
    # 轻量稳定性惩罚（大幅削弱，目标 < progress 的 10%）
    # ═══════════════════════════════════════════
    # 只保留角速度惩罚，抑制疯狂旋转
    stability_penalty = -0.005 * abs(ang_vel)

    # ═══════════════════════════════════════════
    # 总奖励
    # ═══════════════════════════════════════════
    w_progress = 5.0
    w_soft     = 1.0
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

### 预期效果

| 指标 | iter 1（旧） | 本轮预期 |
|---|---|---|
| progress_delta mean | 0.016 | ~0.02（clip 不改变均值） |
| stability mean abs | 0.142（9x progress） | ~0.005（~25% progress） |
| soft_landing nonzero_rate | 0.5%（二值） | ~5-15%（连续 3 因子） |

稳定性惩罚从 progress 的 9 倍降到约 25%，soft_landing 从稀疏二值变成连续梯度——两者都应该显著改善学习信号质量。