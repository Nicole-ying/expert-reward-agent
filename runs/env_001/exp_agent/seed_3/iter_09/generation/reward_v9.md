## 诊断

### 1. Agent 发生了什么？
Episode 长（795.9）且得分中等（94.8），没有 crash。但对比 iter 3 的 197.22（progress_delta 骨架），当前得分只有不到一半。Agent 在原点附近徘徊但没有高效完成任务。

### 2. 哪个组件是主要原因？
`distance_reward` 和 `proximity_bonus` 互相内耗。两者都在鼓励靠近原点，但一个是负信号（-d），一个是正信号（2/(1+5d)）。在 d > 0.55 时净效应为负——agent 在远处每步都在受罚，产生"反正都是负的"的习得性无助。`proximity_bonus` 虽然始终为正，但在远处太弱（d=2 时仅 0.18），无法独自提供足够的远程引导。

### 3. 我之前改了什么？
上一轮把稀疏的 `landing_bonus`（nonzero_rate 极低导致 crash）换成了密集的 `proximity_bonus`，成功解决了 crash（-112→94.8）。但 `distance_reward` 的负信号和 `proximity_bonus` 的正信号互相抵消，导致净引导力太弱。

### 修订方案
**用 progress_delta 替代 distance_reward。** `progress_delta = old_dist - new_dist` 提供方向性梯度——靠近原点得正奖励，远离得负惩罚，不动得零。它与 `proximity_bonus` 互补：远处 progress_delta 主导（移动量大），近处 proximity_bonus 主导（位置吸引力强）。系数 5.0 使 progress_delta 在远距离时量级约 0.5–2.5，足以提供清晰的远程引导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步距离和下一步距离
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 主信号1：progress_delta——替代 distance_reward
    # 靠近原点 → 正奖励，远离 → 负惩罚，提供密集的方向性梯度
    # 系数 5.0：远距离时主导（d=2→1.5，delta=0.5，reward=2.5）
    progress_delta_reward = 5.0 * (old_dist - new_dist)

    # 主信号2：proximity_bonus——保持不变
    # 近距离时提供强吸引力（d=0 时=2.0），远距离时弱但始终为正
    proximity_bonus = 2.0 / (1.0 + 5.0 * new_dist)

    # 稳定约束——保持上一轮系数不变
    vel_penalty = 0.1 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.2 * abs(next_obs[4])
    angvel_penalty = 0.05 * abs(next_obs[5])
    stability_penalty = -vel_penalty - angle_penalty - angvel_penalty

    total_reward = progress_delta_reward + proximity_bonus + stability_penalty

    components = {
        'progress_delta_reward': progress_delta_reward,
        'proximity_bonus': proximity_bonus,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```