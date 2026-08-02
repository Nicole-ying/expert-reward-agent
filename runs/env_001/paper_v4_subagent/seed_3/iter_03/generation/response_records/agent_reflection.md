# Response Record

# 设计理由

第 2 轮加入了 `descending_penalty` 后，agent 活到了 1000 步，但 **`landing_bonus` 从未触发**（active_rate 0%），导致 agent 学会悬停/水平巡航以获得 `progress` 奖励，却不肯下降——因为下降没有奖励，且快速下降会被惩罚。  
本轮将僵尸组件 `landing_bonus`（二值触发 10.0）**结构变换为连续、有界、不塌缩的 `landing_approach` 因子**：采用三个子条件的算术平均，而非乘积，避免任何单项为 0 时整体奖励归零。这样 agent 在接近地面、姿态良好、垂速温和时都能获得渐进正反馈，从而被引导向最终着陆状态。

**数学形式**：  
- 地面接近度：`1/(1+ny²)`，y 越小越接近地面 (假设 pad 在 y=0)。  
- 姿态因子：`max(0, 1 - |angle|/0.3)`，阈值 0.3 rad。  
- 垂速因子：`max(0, 1 - |vy|/0.3)`，阈值 0.3 m/s。  
- 算术平均后乘以 `weight=0.05`，使每步最大奖励约为 0.05，远小于 `progress` 的每步平均值（约 0.008）的 2 倍约束被刻意放宽，以便着陆引导信号足够强，但通过小系数控制总量，防止掩盖稳定性约束。

其他组件保持不变，本轮只修改一个组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observation indices (same as before)
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle, n_ang_vel = next_obs[4], next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # ---------- 1. Main progress: distance to target ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new
    w_progress = 10.0

    # ---------- 2. Stability ----------
    w_angle = 0.5
    w_angvel = 0.1
    stability = -w_angle * (n_angle ** 2) - w_angvel * (n_ang_vel ** 2)

    # ---------- 3. Lateral drift ----------
    w_lat = 0.1
    lateral_drift = -w_lat * (nvx ** 2)

    # ---------- 4. Landing approach (continuous, replaces sparse landing_bonus) ----------
    # Ground proximity: y near 0 (assumed pad altitude) gives high factor
    ground_prox = 1.0 / (1.0 + ny ** 2)
    # Attitude factor: angle within 0.3 rad
    angle_factor = max(0.0, 1.0 - abs(n_angle) / 0.3)
    # Vertical speed factor: gentle vertical speed
    vy_factor = max(0.0, 1.0 - abs(nvy) / 0.3)
    landing_factor = (ground_prox + angle_factor + vy_factor) / 3.0
    w_landing = 0.05
    landing_approach = w_landing * landing_factor

    # ---------- 5. Descending penalty ----------
    safe_vy = -0.3
    w_desc = 0.5
    descending_penalty = w_desc * min(0.0, nvy - safe_vy)

    # Combine
    total_reward = (w_progress * progress
                    + stability
                    + lateral_drift
                    + landing_approach
                    + descending_penalty)

    components = {
        "progress": w_progress * progress,
        "stability_penalty": stability,
        "lateral_drift_penalty": lateral_drift,
        "landing_approach": landing_approach,
        "descending_penalty": descending_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: `landing_bonus` 为僵尸组件（0%），所有 episode 均 truncated；信号缺失在于缺少引导下降的连续奖励，而非观测缺失。
- **behavior**: agent 学会悬停/水平巡航以获取 progress 奖励，完全避免下降，从不触发着陆。
- **signal**: 缺失鼓励接近地面并保持姿态的渐进信号；原有的二值 `landing_bonus` 过于稀疏。
- **level**: Level 2 — 结构变换（稀疏二值 → 连续 bounded factor，算术平均避免塌缩）。
- **hypothesis**: 连续着陆因子会在降落过程中提供梯度信号，引导 agent 降低高度并维持姿态，最终触发真实着陆。
- **risk**: 若系数过大，agent 可能为了获取高额着陆奖励而放弃稳定性，导致坠毁；后续可能需要调整 `weight` 或增强稳定性惩罚。
