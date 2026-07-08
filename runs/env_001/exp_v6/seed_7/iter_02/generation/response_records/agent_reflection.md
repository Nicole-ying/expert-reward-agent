# Response Record

## 诊断

### 1. Agent 发生了什么？
**Crash。** episode_length=68，100% early terminal。Agent 在快速坠落/撞毁，每一轮都撑不到 150 步。

### 2. 哪个组件是主因？
**stability_penalty。** ratio_to_progress = **-4.24**，远超安全阈值 0.5。这意味着：agent 每靠近目标一步获得 +0.016，但同时因为有任何速度/角度就受到 -0.068 的惩罚。净效果是「动就亏」，agent 没有动力朝目标飞行，反而倾向于不动——但在 Lunar Lander 中不动 = 自由落体 = 撞毁。

progress_delta 本身均值 0.016 是合理的步级信号，但被 4 倍大的惩罚完全淹没。

### 3. 我之前改了什么？
这是第 1 轮，无历史。问题出在初始系数设置：速度惩罚 0.05、角度惩罚 0.1 在 Lunar Lander 的尺度下太重了。

### 修复方案
**层次 1：改系数。** 将 stability_penalty 各系数降低约 10 倍，使 ratio 降到 0.5 以下。同时注意到 landing_bonus 触发率仅 0.2%，几乎等于死信号，但本轮先不动它（原则 3：一次改一个信号）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward —— 每一步靠近目标的正向奖励
    dist_current = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_current - dist_next  # 正值表示靠近

    # 稳定与安全约束：系数降低约10倍，使 penalty 不再主导 progress
    # 原系数 0.05/0.1/0.01 → ratio -4.24，agent 不敢动
    # 新系数 0.005/0.01/0.001 → 预期 ratio ~0.4，低于 0.5 安全线
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    stability_penalty = -0.005 * (abs(vel_x) + abs(vel_y)) - 0.01 * abs(angle) - 0.001 * abs(angular_vel)

    # 软着陆 proxy：当接近目标、低速、姿态小且双腿接触时给予一次性小奖励
    landing_bonus = 0.0
    left_contact = next_obs[6] > 0.5
    right_contact = next_obs[7] > 0.5
    if (dist_next < 0.1 and
        (vel_x**2 + vel_y**2) ** 0.5 < 0.2 and
        abs(angle) < 0.2 and
        left_contact and right_contact):
        landing_bonus = 1.0

    total_reward = progress_delta + stability_penalty + landing_bonus

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

### 改动说明
| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| 速度惩罚系数 | 0.05 | 0.005 | 10x 降低，ratio 从 4.24 → ~0.42 |
| 角度惩罚系数 | 0.10 | 0.01 | 10x 降低 |
| 角速度惩罚系数 | 0.01 | 0.001 | 10x 降低 |

只改了一个组件（stability_penalty 的三个系数），其他所有逻辑和 landing_bonus 保持不变。下一轮反馈可以清楚归因：如果 ratio 降到 0.5 以下且 episode 变长，说明方向正确；如果 penalty 仍然过高，再考虑改数学形式（如距离门控）。
