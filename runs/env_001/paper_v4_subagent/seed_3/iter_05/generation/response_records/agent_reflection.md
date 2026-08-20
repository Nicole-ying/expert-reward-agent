# Response Record

# 设计理由
当前 reward 已超额完成目标分数（210.4），但 episode 平均长度仍有 655 步，其中 45% 为超时未着陆。在现有信号齐全的前提下，agent 倾向在接近阶段保守平移或小幅徘徊，导致着陆时刻偏晚。为提高“尽快降落”的激励，本修改将主进度奖励权重 `w_progress` 从 **10.0 提升至 15.0**，使接近目标的正向信号更强烈，有望缩短接近时间，同时保留原有的着陆姿态引导与成功奖励，避免损害最终着陆质量。这是系数调优的 Level 1 改动，不改变函数结构，风险可控。

（per‑step 惩罚合计约 0.021，拟议提升后主信号 per‑step 约 0.10，比率 0.21 < 0.5，满足设计校准。)

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observation indices
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
    w_progress = 15.0   # ← 唯一修改：提升接近速度激励

    # ---------- 2. Stability ----------
    w_angle = 0.5
    w_angvel = 0.1
    stability = -w_angle * (n_angle ** 2) - w_angvel * (n_ang_vel ** 2)

    # ---------- 3. Lateral drift ----------
    w_lat = 0.1
    lateral_drift = -w_lat * (nvx ** 2)

    # ---------- 4. Landing approach ----------
    ground_prox = 1.0 / (1.0 + ny ** 2)
    angle_factor = max(0.0, 1.0 - abs(n_angle) / 0.3)
    vy_factor = max(0.0, 1.0 - abs(nvy) / 0.3)
    landing_factor = (ground_prox + angle_factor + vy_factor) / 3.0
    w_landing = 0.05
    landing_approach = w_landing * landing_factor

    # ---------- 5. Success bonus ----------
    w_success = 0.1
    on_ground = max(n_left, n_right)               # 0 or 1
    close_to_target = 1.0 if dist_new < 0.2 else 0.0
    success_bonus = w_success * on_ground * close_to_target

    # Combine
    total_reward = (w_progress * progress
                    + stability
                    + lateral_drift
                    + landing_approach
                    + success_bonus)

    components = {
        "progress": w_progress * progress,
        "stability_penalty": stability,
        "lateral_drift_penalty": lateral_drift,
        "landing_approach": landing_approach,
        "success_bonus": success_bonus
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 观测维度全部利用，无信号缺口，无僵尸组件；当前 agent 已能稳定着陆，目标已解，但仍有部分 episode 超时。
- **behavior**: agent 在大多数回合成功着陆并累积奖励，但在接近目标阶段存在不必要的徘徊，导致平均长度 655 步、45% 回合超时。
- **signal**: 主进度信号偏弱（原权重 10.0），对“快速接近”激励不足；着陆引导和成功奖励功能正常。
- **level**: Level 1 — 尺度修复（系数调优）
- **hypothesis**: 提升进度系数将增强接近目标的梯度，使 agent 减少空中犹豫，更快抵达着陆触发区域，从而缩短 episode 长度同时维持着陆质量。
- **risk**: 若速度激励过强，可能略微牺牲姿态稳定性，但现有 `stability_penalty`、`landing_approach` 及 `success_bonus` 仍会约束着陆质量，整体性能预计不会倒退。
