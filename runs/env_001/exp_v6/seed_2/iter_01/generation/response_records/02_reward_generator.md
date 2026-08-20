# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty (anchor, tiny weight) ----
    distance_penalty = -0.05 * d_next

    # ---- stability penalty (small weight) ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.05 * abs(vx)
        - 0.05 * abs(vy)
        - 0.2 * abs(angle)
        - 0.2 * abs(ang_vel)
    )

    # ---- soft landing proxy (first double-contact landing) ----
    bonus = 0.0
    # Constants – tunable thresholds
    LANDING_DIST_THRESH = 0.5
    VEL_THRESH = 0.2
    ANGLE_THRESH = 0.1

    # New double contact?
    new_contact = (
        next_obs[6] == 1.0 and next_obs[7] == 1.0 and
        (obs[6] != 1.0 or obs[7] != 1.0)
    )
    if new_contact:
        if d_next < LANDING_DIST_THRESH:
            if abs(vx) < VEL_THRESH and abs(vy) < VEL_THRESH:
                if abs(angle) < ANGLE_THRESH:
                    bonus = 1.0

    # ---- total ----
    total = progress_delta + distance_penalty + stability_penalty + bonus

    components = {
        "progress_delta_reward": progress_delta,
        "distance_penalty": distance_penalty,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": bonus,
        "total_reward": total
    }
    return float(total), components
```

# reward_v1 设计说明

- **组件及角色**
  - `progress_delta_reward`（主学习信号）：每一步更靠近目标垫时获得正奖励，驱动飞行器持续向中心移动。
  - `distance_penalty`（小权重锚）：对当前距离施加微小负奖励，防止在目标附近“驻足”而不完成着陆，同时不干扰 `progress_delta` 的主导地位。
  - `stability_penalty`（轻量稳定约束）：惩罚水平/竖直速度、机体倾角与角速度，鼓励平稳飞行与接近目标时的减速、竖直姿态。
  - `soft_landing_bonus`（任务完成 proxy）：仅在 *首次同时* 获得左右支撑腿接触、且满足近距、低速、小倾角条件时给予一次性奖励，引导完成软着陆而不伪造成功标志。

- **未使用 terminal_success_reward / terminal_failure_penalty**  
  环境卡片明确 `explicit_success_flag_available=false` 且 `info` 为空，不存在任何成功/失败标志。因此 v1 中不使用任何需要显式标志的终点奖励/惩罚，避免奖励函数“幻想”不存在的信号。

- **后续迭代预留**
  - `energy_penalty`：待飞行器能够稳定接近并着陆后，再引入动作推力惩罚以优化燃料效率。
  - `time_penalty`：若出现长时间徘徊不前的情况，后续可加入微小的时间惩罚。
  - `gated_reward` / `dynamic curriculum`：当前基础组件已覆盖主要学习信号，这些高级成分留待后续版本加入。
  - 阈值调整：`LANDING_DIST_THRESH`、`VEL_THRESH`、`ANGLE_THRESH` 将在训练中根据实际行为观测进一步校准。

- **需关注的 failure mode**
  - **目标附近震荡**：主要由 `progress_delta` 与 `distance_penalty` 的结构可能导致在小距离内反复进退，需观察奖励是否收敛、以及 `soft_landing_bonus` 是否被触发。
  - **高速撞击**：若稳定性惩罚过弱，飞行器可能
