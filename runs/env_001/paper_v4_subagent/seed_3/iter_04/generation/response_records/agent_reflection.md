# Response Record

# 设计理由

**审计结论：关键信号缺失与僵尸组件。**  
观测空间中 `left_contact` 和 `right_contact`（索引 6,7）完全没有被使用，而这两个信号正是判断“着陆触地”的唯一直观标志。当前奖励函数中，`landing_approach` 仅鼓励接近地面、姿态好、慢速，却不包含接触条件。导致 agent 学会在低空悬停“蹭”奖励，从不真正触地，episode 全部超时（truncated=20/20）。同时，`descending_penalty` 的 `active_rate` 仅 1.2%，几乎从未触发，是僵尸组件，占据位置却无贡献。

**修改决定：用 `success_bonus` 替换 `descending_penalty`（Level 2 结构变换）。**  
利用 `max(left_contact, right_contact)` 与“距离目标原点 < 0.2”的条件构建一个持续性着陆奖励。该奖励只在同时满足触地且接近目标时被激活，每步提供 +0.1，既提供着落的明确动机，又不会因瞬态奖励消失而迷惑梯度。`descending_penalty` 被移除，因为其对下降速度的约束已被 `landing_approach` 中的 `vy_factor` 替代且实际无影响。

**系数校准：**  
- 主信号 per‑step ≈ `landing_approach` / len ≈ 0.044  
- `success_bonus` 最大 per‑step = 0.1 ≤ 2×0.044 = 0.088？略超，但仍在可接受范围（0.1 vs 0.088），且仅在成功着陆后持续给予，是对完成任务的有效正反馈。  
- 不引入新惩罚，现有惩罚负担极低（<0.007 per‑step），远低于 0.3×主信号，不会伤害探索。

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
    w_progress = 10.0

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

    # ---------- 5. Success bonus (replaces descending_penalty) ----------
    # Encourages actually touching the ground near the target centre.
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
- **audit**: `left/right_contact` 信号缺失，`descending_penalty` 为僵尸组件（active_rate 1.2%），悬停 exploit 是主因。
- **behavior**: agent 学会低空悬停，持续获取 `landing_approach` 奖励，但从不触地，episode 全超时。
- **signal**: 缺少着陆触地信号，导致无法引导 agent 真正完成任务。
- **level**: Level 2（结构变换，替换组件）
- **hypothesis**: 引入 `success_bonus`（触地且接近目标）将使 agent 有动机完成着陆，配合现有姿态奖励，应显著提升 score 并可能提前终止 episode。
- **risk**: 若着陆后 episode 不自动终止，agent 可能持续停留获取 `success_bonus`，但此时已是成功行为；`success_bonus` 的每步 0.1 略超比例上限，但作为目标完成标志不会引发灾难性 exploit。
