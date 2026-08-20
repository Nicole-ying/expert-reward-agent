# Response Record

## 诊断

### 1. Agent 发生了什么？

- **分数跃升**：从 iter 1 的 -110（70步 crash）到 iter 2 的 156（916步），stability_penalty 削弱是成功的。
- **但 agent 在 exploit**：episode_length=916 接近上限（~1000），说明 agent 没有真正完成着陆——它在原点附近"磨蹭"收集 soft_landing_proxy 奖励。nonzero_rate=61% 意味着超过一半的步数都在触发 landing 奖励，这不正常——真正的着陆应该是 episode 末尾的稀疏事件。

### 2. 根因

- **soft_landing_proxy ratio=24x**，是 progress_delta_reward 的 24 倍。一旦 agent 到达原点附近，progress 信号趋近于零（距离变化极小），而 soft_landing_proxy 每步给 0.1，成为唯一驱动力。
- **条件太宽松**：`dist<0.2`、`speed<0.1`、`angle<0.05` 三个阈值让 agent 在原点附近"晃悠"就能持续触发，不需要真正稳定着陆。

### 3. 上轮改动回顾

上一轮削弱了 stability_penalty（系数 ×0.1），效果显著。soft_landing_proxy 上轮未改动，本轮聚焦它。

### 修改策略

**层次 1：收紧 soft_landing_proxy 的触发阈值**。不改变数学形式（仍是二值条件），但把三个阈值收紧，让 nonzero_rate 从 61% 降到合理区间（期望 10-25%），从而降低 ratio。reward 值保持 0.1 不变——它是稀疏但有意义的终端信号。

- `dist < 0.2` → `dist < 0.12`
- `speed < 0.1` → `speed < 0.05`
- `angle < 0.05` → `angle < 0.03`

其他组件保持不变。下一轮观察 nonzero_rate 和 ratio 变化。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract states ----------
    # obs
    x, y = obs[0], obs[1]
    # next_obs
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 1. main learning signal: progress toward (0,0) ----------
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress_delta_reward = dist_obs - dist_next

    # ---------- 2. stability / smoothness penalty ----------
    vel_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    ang_vel_penalty = abs(ang_vel)

    w_vel = 0.001
    w_angle = 0.005
    w_angvel = 0.001
    stability_penalty = - (w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy ----------
    # DIAGNOSIS: nonzero_rate=61%, ratio_to_progress=24x — agent exploits loose thresholds.
    # FIX: tighten all three thresholds to make this a rarer, more meaningful signal.
    #   dist: 0.20 -> 0.12, speed: 0.10 -> 0.05, angle: 0.05 -> 0.03
    near_target = dist_next < 0.12
    low_speed = abs(vx) < 0.05 and abs(vy) < 0.05
    upright = abs(angle) < 0.03
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)

    soft_landing_proxy = 0.1 if (near_target and low_speed and upright and both_contact) else 0.0

    # ---------- total ----------
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy

    # ---------- components dict ----------
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```
