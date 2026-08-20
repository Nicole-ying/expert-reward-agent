# Response Record

## 诊断

**症状分析：**
- 100% 早停（70 步），score=-112.57 → agent 在 crash，根本没学会靠近目标。
- `stability_penalty` ratio_to_progress = **-0.85**，远超 0.5 危险线。这意味着稳定性惩罚几乎完全抵消了 progress 信号，`total_reward` 均值仅 0.0024——agent 每步得到的净学习信号趋近于零。
- `progress_delta` 本身也很小（0.016/步），但根因是惩罚太强，不是 progress 太弱。progress 至少是正向的、有梯度的。

**根因判断：** stability_penalty 权重过大，把 progress 信号淹没。Agent 在"动就挨罚"和"不动没奖励"之间选择了崩溃——因为没有有效的正向引导。

**上一轮动作：** 这是第一轮，无历史对照。直接修 penalty 量级。

**修改方案：** 将 stability_penalty 权重降低 10 倍（层次 1：改系数），使 ratio_to_progress 从 -0.85 降到约 -0.08，远低于 0.5 危险阈值。这是本轮唯一改动。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- main progress signal: reduction in Euclidean distance to target ----
    # target position is (0, 0) in the relative coordinate system
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5
    progress_delta = dist_obs - dist_next   # positive when moving closer

    # ---- light stability penalty on next observation ----
    # penalise linear velocity, body angle and angular velocity
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # weights reduced 10x from previous iteration (were 0.01, 0.01, 0.005)
    # previous ratio_to_progress was -0.85 — penalty dominated progress entirely
    w_vel = 0.001
    w_angle = 0.001
    w_angvel = 0.0005

    stability_penalty = (
        - w_vel * (abs(vx) + abs(vy))
        - w_angle * abs(angle)
        - w_angvel * abs(ang_vel)
    )

    # ---- total reward ----
    total_reward = progress_delta + stability_penalty

    # ---- component logging ----
    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```
