# Response Record

## 诊断

**症状：** episode 平均 70 步，全部提前终止（crash），score=-110.96。Agent 在快速坠毁。

**根因分析：**

| 信号 | ratio_to_progress | 问题 |
|------|------------------|------|
| `stability_penalty` | **-0.903** | 惩罚几乎完全抵消 progress 信号 |
| `soft_landing_proxy` | 0.025 | nonzero_rate=0.4%，等于摆设 |
| `total_reward` | 0.122 | 均值仅 0.002/步，agent 几乎收不到梯度 |

核心问题：`stability_penalty` 的 magnitude 达到 progress 信号的 90%，导致每步净奖励接近零。Agent 分不清好坏动作 → 随机漂移 → crash。`soft_landing_proxy` 触发率 0.4% 也无法提供终端引导。

**本轮行动：** 降低 stability_penalty 系数 10 倍（层次 1）。这是本轮唯一改动，让下轮可归因。soft_landing 的二值形态问题留到下一轮。

---

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
    # DIAGNOSIS: ratio_to_progress was -0.903, penalty nearly cancels progress signal.
    # FIX: reduce all coefficients by 10x to bring ratio well under 0.5.
    vel_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    ang_vel_penalty = abs(ang_vel)

    w_vel = 0.001      # was 0.01
    w_angle = 0.005     # was 0.05
    w_angvel = 0.001    # was 0.01
    stability_penalty = - (w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy (unchanged this round) ----------
    near_target = dist_next < 0.2
    low_speed = abs(vx) < 0.1 and abs(vy) < 0.1
    upright = abs(angle) < 0.05
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
