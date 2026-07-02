# Response Record

## 诊断

**症状分析：**
- Episode 长度 99.8，90% 提前终止 → agent 仍然在 crash
- progress_delta 均值仅 0.015，信号极弱 → agent 几乎不移动
- stability_penalty ratio = -0.098，量级已健康 → 不再主导，但也没教会稳定性
- original_env_reward = -0.808 → 环境因 crash 给出大量负反馈

**根因判断：**
上一轮把 stability penalty 削弱 10 倍后，ratio 从 -0.85 降到 -0.098，解决了"惩罚主导"问题。但 agent 仍然 crash。真正的问题是：**稳定性惩罚均匀施加在所有距离上**——agent 远离目标时需要大幅机动来学习控制，但惩罚抑制了速度，让它不敢动也学不会。与此同时 progress 信号太弱（0.015/步），无法提供足够的梯度驱动探索。

**本轮策略（一个改动）：距离门控稳定性惩罚。**
用 `gate = 1/(1+dist)` 让稳定性惩罚随距离自动缩放：远处 gate≈0，agent 自由机动；近处 gate→1，惩罚生效，迫使精细控制。这是一个自然的课程学习机制。惩罚的 base 权重恢复到 iter 1 水平（距离门控会自动削弱远处惩罚），使近目标区域有足够的稳定性引导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- progress signal: reduction in Euclidean distance to target ----
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5
    progress_delta = dist_obs - dist_next   # >0 when moving toward target

    # ---- distance-gated stability penalty ----
    # Gate: 1/(1+dist) → near 0 when far (free to maneuver),
    #                     near 1 when at target (enforce fine stability).
    # This creates a natural curriculum: aggressive far, careful near.
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    gate = 1.0 / (1.0 + dist_next)

    # Base weights at iter-1 levels; distance gating auto-scales them down
    # when far from target, preventing the penalty from dominating progress.
    w_vel = 0.01
    w_angle = 0.01
    w_angvel = 0.005

    stability_penalty = -gate * (
        w_vel * (abs(vx) + abs(vy))
        + w_angle * abs(angle)
        + w_angvel * abs(ang_vel)
    )

    # ---- total reward ----
    total_reward = progress_delta + stability_penalty

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```
