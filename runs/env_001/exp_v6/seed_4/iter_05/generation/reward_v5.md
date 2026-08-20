## 诊断

### 1. Agent 发生了什么？
episode_length=71.3，100% early terminal。Agent 在 crash——每次都在远早于 150 步时终止且得分极差。

### 2. 哪个组件是主要原因？
核心问题：**stability_penalty 主导了 shaping_reward**。

| 信号 | mean | ratio_to_progress |
|------|------|-------------------|
| shaping_reward | +0.026 | 基准 |
| stability_penalty | -0.068 | **2.6x**（超过主信号！） |
| **total_reward** | **-0.042** | 净负 |

按原则 1，惩罚项均值超过主信号 50% 时 agent 会选择"不动"。这里 stability_penalty 是 shaping 的 2.6 倍，agent 无论怎么做都被净惩罚——连"存在"都要被扣分。它学不到任何有效策略，只能 crash。

另外，gamma=0.99 造成静止时也有 `(1-γ)×dist_current` 的正奖励，给 agent 留了"不动也能得分"的后门（虽然被 stability_penalty 抵消了）。

### 3. 我之前改了什么？
上一轮（iter 4）从 progress+soft_landing+stability 换成了 potential_shaping+stability。骨架换了但 stability 系数没动，导致惩罚仍然压倒塑形信号。这次聚焦**削弱 stability_penalty** 并**修正 gamma**。

### 验证错误修复
上一版 `components` 中有两个问题：① key `potential_shaping_reward` 与公式中变量名 `shaping_reward` 不匹配；② `total_reward` 是返回值而非公式中的组件，不应出现在 components 中。

## 修改方案

- **层次 1（改系数）**：stability_penalty 三系数各降 10x（0.05→0.005, 0.02→0.002, 0.05→0.005），使其均值降至 ~0.007，约为 shaping 的 27%，远低于 50% 红线。
- **层次 1（改系数）**：gamma 从 0.99 改为 1.0，消除静止时的虚假正奖励，shaping 仅在真正移动靠近目标时才为正。
- **修复 components**：key 与公式变量名一致，移除 `total_reward`。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 诊断与修改理由 ===
    # 问题：stability_penalty (-0.068) 是 shaping_reward (0.026) 的 2.6 倍，
    #       agent 在任何状态下都被净惩罚，导致 episode 短 (71步)、全部 crash。
    # 修改1：stability 系数降低 10x，使其成为弱背景信号（~27% of shaping）。
    # 修改2：gamma 1.0 消除静止时的 (1-γ)*dist 虚假奖励。
    # 修改3：components key 与公式变量名一致，移除 total_reward。

    # 1. 势能塑形：Φ = -distance, gamma = 1.0
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    # gamma=1.0 → shaping = dist_current - dist_next，纯进度信号
    shaping_reward = dist_current - dist_next

    # 2. 稳定性惩罚：轻量背景约束
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # 系数降低 10x：原 0.05/0.02/0.05 → 现 0.005/0.002/0.005
    w_vel = 0.005
    w_ang = 0.002
    w_angle = 0.005

    stability_penalty = -(
        w_vel * (abs(vel_x) + abs(vel_y)) +
        w_ang * abs(ang_vel) +
        w_angle * abs(angle)
    )

    total_reward = shaping_reward + stability_penalty

    # components 只放总公式中直接出现的变量，key 与变量名一致
    components = {
        "shaping_reward": shaping_reward,
        "stability_penalty": stability_penalty,
    }
    return float(total_reward), components
```