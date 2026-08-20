# Response Record

## 诊断

### 1. 验证失败原因
上一版代码的 `components` 字典中包含了 `"total_reward": total_reward`，违反约束「components 只放总公式中直接出现的变量，**不包含 total_reward**」。

### 2. 训练表现诊断

| 信号 | 当前值 | 判断 |
|------|--------|------|
| `soft_landing_proxy` | ratio 6.12x, nonzero 48.2% | 工作正常，但比 best (iter 3: 0.294) 弱，饱和 `x/(1+x)` 压低了信号 |
| `stability_penalty` | nonzero 0.35%, mean ≈ 0 | **已死**。`late_weight` 调度 + 极小系数让它从未生效 |
| `original_env_reward` | ratio -7.2 (与 total 符号相反) | 持续 misaligned（环境罚燃料，我们奖靠近），但这在 Landing 环境中是结构性的——燃料惩罚是背景噪声，不是失败原因 |
| score 137.61 vs best 169.76 | -19% | 明显低于 best，核心差异是 proxy 被饱和削弱了 ~30% |

### 3. 本轮修改方案

**Revert 方向**：回到 best (iter 3) 的 raw product proxy 路线（无饱和），因为饱和实验证明有害。

**新修改**（满足 revert 规则"不能原样复制"）：
- 砍掉已死的 `stability_penalty`，替换为一个轻量的 **`center_bonus`**——在目标附近（半径 0.5）给予连续正信号。这不同于 progress（delta 方向信号），它提供"留在目标附近"的绝对位置奖励，天然引导稳定悬停/降落，且不会像 penalty 那样抑制探索。
- 修复验证错误：从 components 中移除 `total_reward`。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    cx, cy = float(obs[0]), float(obs[1])
    nx, ny = float(next_obs[0]), float(next_obs[1])

    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标（scale=8，骨架推荐 5~20）
    progress_reward = (d_curr - d_next) * 8.0

    # 2. 软着陆近似信号：连续乘积（无饱和，回到 best 路线）
    vx, vy = float(next_obs[2]), float(next_obs[3])
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(float(next_obs[4]))

    proximity_factor = max(0.0, 1.0 - d_next / 0.3)
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    angle_factor = max(0.0, 1.0 - angle / 0.3)
    contact_factor = (
        max(0.0, float(next_obs[6])) * max(0.0, float(next_obs[7]))
    )

    soft_landing_proxy = (
        proximity_factor * speed_factor * angle_factor * contact_factor
    )

    # 3. 中心邻近奖励：替代已死的 stability_penalty
    #    半径 0.5 内给予正向信号（最大 0.2），引导 agent 留在目标附近
    #    比 penalty 更友好——鼓励靠近而非惩罚探索
    center_bonus = 0.2 * max(0.0, 1.0 - d_next / 0.5)

    total_reward = progress_reward + soft_landing_proxy + center_bonus

    components = {
        "progress_reward": progress_reward,
        "soft_landing_proxy": soft_landing_proxy,
        "center_bonus": center_bonus,
    }
    return float(total_reward), components
```
