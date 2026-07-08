# Response Record

## 诊断

### 1. Agent 状态
Score=249（历史最佳），episode_length=276，无 crash，无 early termination。表现良好，已标记 `target_solved_new_best`。

### 2. 组件分析

| 组件 | mean | ratio_to_progress | 判断 |
|------|------|------------------|------|
| progress_delta_reward | 0.003 | 1.0x | **太弱**。骨架文档建议 scale 5~20，当前 scale=1 只有 0.003/step |
| soft_landing_proxy | 0.206 | 73.5x | 主导总奖励。但 nonzero_rate=69.7%，在目标附近提供有效引导，形态本身没问题 |
| stability_penalty | -0.0007 | -0.25x | 可忽略。但占 progress 的 23%，骨架建议 <10% |

**根因**：landing proxy 在 dist<0.5 范围内工作良好，但在剩余 30% 的 step（agent 离目标较远时）完全为零，此时 progress 信号仅 0.003/step，几乎无法提供有效梯度。

### 3. 上一轮动作回顾
Iter 4→5：把 soft_landing_proxy 从 `1/(1+kx)` 四因子乘积改为 `max(0,1-x/D)` 形式，系数 0.4。效果极好（-107 → 249）。landing proxy 的数学形态已经正确，不需要再动。

### 修改决策
**层次 1：改系数**。将 progress_delta_reward 乘以 scale=5，使其 mean 从 0.003 → ~0.015，ratio 从 73x → ~14x。这同时满足骨架的两个建议：scale 在 5~20 范围，且 stability_penalty 会被压低到 progress 的 ~5%（<10%）。landing proxy 保持不动，因为它的形态和引导效果已经验证有效。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract states ----------
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 1. main learning signal: progress toward (0,0) ----------
    # DIAGNOSIS: scale=1 gave mean=0.003, too weak vs landing proxy (73x).
    # Skeleton recommends scale 5~20. Using 5 brings mean to ~0.015,
    # ratio to ~14x, and pushes stability_penalty below 10% of progress.
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress_scale = 5.0
    progress_delta_reward = progress_scale * (dist_obs - dist_next)

    # ---------- 2. stability / smoothness penalty ----------
    vel_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    ang_vel_penalty = abs(ang_vel)

    w_vel = 0.001
    w_angle = 0.005
    w_angvel = 0.001
    stability_penalty = -(w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy (CONTINUOUS, max(0,1-x/D) form) ----------
    # distance factor: 1.0 at dist=0, 0.0 at dist>=0.5
    dist_factor = max(0.0, 1.0 - dist_next / 0.5)

    # speed factor: 1.0 at zero speed, 0.0 at total_speed>=0.5
    total_speed = abs(vx) + abs(vy)
    speed_factor = max(0.0, 1.0 - total_speed / 0.5)

    # angle factor: 1.0 at angle=0, 0.0 at |angle|>=0.2
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.2)

    # contact: soft factor [0.33, 1.0], rewards any contact
    contact_factor = 0.33 + 0.335 * (left_contact + right_contact)

    soft_landing_proxy = 0.4 * dist_factor * speed_factor * angle_factor * contact_factor

    # ---------- total ----------
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward,
    }

    return float(total_reward), components
```
