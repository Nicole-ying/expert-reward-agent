## 诊断

### 1. Agent 发生了什么？
**Crash**。episode 长度仅 72 步，100% 早停（<150 步且 score < -50）。agent 在 72 步内就坠毁了。

### 2. 哪个组件是主要原因？

| 组件 | ratio_to_progress | 解读 |
|------|:--:|------|
| stability_penalty | **-0.88** | 惩罚达到主信号的 88%，严重越界 |

**原则 1 红线**：惩罚项的绝对值不应超过主信号的 50%。当前 stability_penalty 是 progress 的 88%，agent 面临一个困境——每前进 1 单位得到的奖励，几乎被"动一下就要罚"完全抵消。结果就是 agent 不敢做大幅度机动，推力不足 → 被重力拉向地面 → 坠毁。

soft_landing_proxy 触发率 0.34%（约每 300 步一次，而 episode 只有 72 步），几乎是死代码，但它 ratio 小（11%），不是本轮主要矛盾。

### 3. 上一轮做了什么？
这是第一轮，没有历史包袱。

### 修改方案

**层次 1：削系数。** stability_penalty 的三个系数从 0.01 降到 0.002（降低 5 倍）。预期 penalty 从 -0.014 降到约 -0.003，ratio 从 88% 降到约 18%，远低于 50% 安全线。其他组件不动。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 1. 提取观察量 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    vel_x, vel_y = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 2. 主学习信号：进度差奖励 ──
    dist_old = (x ** 2 + y ** 2) ** 0.5
    dist_new = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_old - dist_new

    # ── 3. 轻量稳定约束（系数从 0.01 → 0.002，降低 5 倍）──
    # 上一轮：penalty ratio = -0.88，严重压制 progress 信号
    # 本轮目标：ratio ≈ -0.18，让 agent 敢于机动
    stability_penalty = -0.002 * (abs(vel_x) + abs(vel_y)) \
                        -0.002 * abs(angle) \
                        -0.002 * abs(angular_vel)

    # ── 4. 软着陆近似信号（本轮不动）──
    near_target = dist_new < 0.15
    low_speed = (vel_x ** 2 + vel_y ** 2) ** 0.5 < 0.2
    upright = abs(angle) < 0.1
    both_legs_down = (left_contact > 0.5) and (right_contact > 0.5)

    soft_landing_proxy = 0.5 if (near_target and low_speed and upright and both_legs_down) else 0.0

    # ── 组合总奖励 ──
    total_reward = progress + stability_penalty + soft_landing_proxy

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```