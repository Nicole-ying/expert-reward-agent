# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：向前速度
    horizontal_speed = next_obs[2]
    progress = 2.0 * horizontal_speed

    # 稳定/安全约束：姿态角度超出健康范围时软惩罚（hinge）
    hull_angle = next_obs[0]
    max_allowed_angle = 0.3  # 约 17 度
    posture_penalty = -5.0 * max(0.0, abs(hull_angle) - max_allowed_angle)

    # 稳定/安全约束：角速度惩罚，抑制剧烈摇晃
    ang_vel = next_obs[1]
    ang_vel_penalty = -0.05 * (ang_vel ** 2)

    # 效率/动作代价：轻微二次惩罚，抑制过大关节力矩
    action_cost = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    total_reward = progress + posture_penalty + ang_vel_penalty + action_cost
    components = {
        'progress_reward': progress,
        'posture_penalty': posture_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'action_cost': action_cost
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **task_family / dynamics_subtype**：locomotion_continuous_control / planar_bipedal_rough_terrain_gait  
  任务画像明确要求最大化前进距离，同时避免摔倒并降低关节能耗。

- **selected reward roles**（基于 environment_card.md 的 reward_role_decomposition）：
  - `forward_progress`：主学习信号（mandatory）
  - `fall_prevention`：稳定/安全约束（mandatory），拆分为姿态 hinge 与角速度二次惩罚
  - `energy_efficiency`：效率/动作代价（mandatory，但权重极低，不压制探索）

- **role_to_signal_mapping 与 formula operator**：
  - `forward_progress` ← `next_obs[2]`（horizontal_speed），采用 **dense_state_signal（线性）**，即 `2.0 * signal`。
  - `fall_prevention`：
    - 姿态部分 ← `next_obs[0]`（hull_angle），采用 **dense_state_signal（hinge）**，在角度超出 0.3 rad 启动惩罚，避免全时压制正常步态。
    - 角速度部分 ← `next_obs[1]`（hull_angular_velocity），采用 **quadratic_penalty**，抑制急剧摇晃。
  - `energy_efficiency` ← `action[0..3]`，采用 **action_efficiency（二次惩罚）**，权重设为 `-0.01 * sum(action_i**2)`。

- **excluded roles 及原因**：
  - `contact_symmetry`：v1 阶段需要先获得基本前进能力，步态对称的额外约束可能干扰探索。
  - `vertical_smoothness`：垂直速度惩罚在历史尝试中已出现，本设计刻意排除以避免抑制必要的垂直运动（如跨越障碍），而只通过姿态/角速度约束保证稳定。
  - `alive_bonus`：会鼓励原地不动，与前进目标冲突。
  - `target_velocity_tracking`：缺少预设目标速度。
  - `lidar_based_terrain_penalty`：难以将高维 LIDAR 映射为标量奖励，且容易引入噪声。

- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**：
  - 环境 info 为空，明确无 success / failure flag，无法依赖终止事件提供硬信号。
  - 摔倒的前兆信号（急剧增大倾角与角速度）已由 `posture_penalty` 和 `ang_vel_penalty` 提供连续梯度，不再额外需要推断式 terminal penalty。

- **留到后续迭代的职责**：
  - `contact_symmetry`（能在步态已稳定后加入）
  - `vertical_smoothness`（可在出现过度跳跃时再轻量使用）
  - 更复杂的 `soft_health_gate` 或 `preview_conditioned_reward`（如果没有 LIDAR 到奖励的简单映射，则暂不引入）

- **训练后预期的 failure modes 与观察点**：
  - agent 可能因动作惩罚过小仍保持高能耗——后续可适当提高 `action_cost` 系数。
  - 如果 agent 倒向一侧但角度未触发 hinge（角度长时间维持在小幅越界附近），可考虑将 hinge 阈值微调。
  - 如果 progress 信号饱和导致 agent 满足于低速巡航，可将 `progress` 改为凸化形式（`horizontal_speed**2`）。
  - 若 agent 频繁摔倒且总和 reward 仍为正，说明主奖励在失败前给予的正向贡献过大，可能需要引入 `soft_health_gate` 在接近危险时衰减 progress。
