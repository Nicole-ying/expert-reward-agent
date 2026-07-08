# Response Record

## 诊断

### 1. Agent 发生了什么？
- Episode 长度 69，远低于 150，全部 20 个 episode 都是 early terminal。
- Score = -88.79，极负。**Agent 在快速 crash**，根本没学到靠近目标。
- `original_env_reward` ratio = -91（与 progress 反向），环境判定 agent 行为是失败的。

### 2. 哪个组件是主因？
- **`stability_penalty`：ratio = -3.63**，是 progress 信号的 3.6 倍，且方向相反。远超安全线（|ratio| < 0.5）。
- 每走一步，agent 被罚 0.058，而 progress 只奖 0.016。agent 动就被罚，不动又拿不到 progress，陷入死局。
- `soft_landing_proxy` nonzero_rate 仅 0.32%，但这是 crash 太早的后果，不是原因。

### 3. 根因：stability_penalty 系数太大，压制了主信号。

### 修改方案：层次 1 — 降低 stability_penalty 系数 10 倍

把 speed/angle/angular_v 系数全部缩小 10 倍，使 ratio 从 -3.63 降到约 -0.36，让 progress 重新成为主导信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，平台中心
    cx, cy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]

    # 当前位置与目标距离
    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    # 下一时刻与目标距离
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标
    progress_reward = d_curr - d_next

    # 2. 轻量稳定约束（系数降低10倍，ratio 从 -3.63 → ~-0.36）
    vx, vy = next_obs[2], next_obs[3]
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(next_obs[4])
    angular_v = abs(next_obs[5])

    stability_penalty = (
        -0.005 * speed    # was -0.05
        - 0.01 * angle    # was -0.1
        - 0.005 * angular_v  # was -0.05
    )

    # 3. 软着陆完成近似信号（保持不变）
    soft_landing_proxy = 0.0
    if (
        d_next < 0.15
        and speed < 0.2
        and angle < 0.1
        and next_obs[6] == 1.0
        and next_obs[7] == 1.0
    ):
        soft_landing_proxy = 1.0

    # 总奖励
    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward,
    }
    return float(total_reward), components
```
