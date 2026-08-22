# Response Record

# reward_v1.py

```python
import math

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander navigation task.
    Roles: goal_distance_progress (improvement_delta of distance to pad),
           soft_landing (proximity‑gated velocity penalty),
           orientation_penalty (quadratic on body angle),
           landing_contact_success (soft completion proxy).
    """
    # ------------------------------------------------------------------
    # Unpack observations
    # ------------------------------------------------------------------
    prev_x, prev_y = obs[0], obs[1]
    prev_vx, prev_vy = obs[2], obs[3]
    # prev_angle = obs[4]          # not used directly
    # prev_angvel = obs[5]        # not used
    # prev_left  = obs[6], prev_right = obs[7]  # contact flags of current step are not needed

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    # next_angvel = next_obs[5]   # not used
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # ------------------------------------------------------------------
    # Distance to target pad  (0,0)
    # ------------------------------------------------------------------
    dist_prev = math.sqrt(prev_x**2 + prev_y**2)
    dist_next = math.sqrt(next_x**2 + next_y**2)

    # ------------------------------------------------------------------
    # 1. Goal distance progress  (improvement_delta)
    #    Positive when moving towards the pad, negative when retreating.
    # ------------------------------------------------------------------
    w_progress = 5.0
    progress = w_progress * (dist_prev - dist_next)

    # ------------------------------------------------------------------
    # 2. Soft landing  (proximity‑gated velocity penalty)
    #    Large penalty for speed when close to the pad, small when far.
    # ------------------------------------------------------------------
    w_vel = 1.0
    k_dist = 20.0   # controls how sharply the gate opens near the pad
    proximity_gate = 1.0 / (1.0 + k_dist * dist_next**2)
    vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate

    # ------------------------------------------------------------------
    # 3. Orientation penalty  (quadratic penalty on tilt)
    #    Body angle should be near zero (upright).
    # ------------------------------------------------------------------
    w_angle = 0.5
    angle_penalty = -w_angle * (next_angle**2)

    # ------------------------------------------------------------------
    # 4. Soft completion proxy  (both legs on pad + very low speed)
    #    Gives a positive boost when the lander appears to have settled.
    # ------------------------------------------------------------------
    w_contact = 2.0
    v_thresh = 0.1
    legs_on_ground = (next_left_contact >= 0.5) and (next_right_contact >= 0.5)
    speed_low = (abs(next_vx) < v_thresh) and (abs(next_vy) < v_thresh)
    contact_bonus = w_contact if (legs_on_ground and speed_low) else 0.0

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = progress + vel_penalty + angle_penalty + contact_bonus

    components = {
        "goal_distance_progress": progress,
        "soft_landing":          vel_penalty,
        "orientation_penalty":   angle_penalty,
        "landing_contact_success": contact_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **任务家族 / 动力学子类型**  
  `navigation_goal_reaching` / `goal_approach_and_soft_contact`（2D 着陆器，离散控制，双腿着陆）。

- **选中的奖励职责**  
  - 主学习信号：`goal_distance_progress`  
  - 稳定/安全约束：`soft_landing`、`orientation_penalty`  
  - 任务完成近似信号：`landing_contact_success`

- **职责 → 信号映射及公式算子**  
  | 职责 | 信号 | 算子 |
  |------|------|------|
  | `goal_distance_progress` | `x_position`, `y_position` → 当前与下一步距离 | `improvement_delta`（期望距离减小） |
  | `soft_landing` | `x_velocity`, `y_velocity` + 距离 | `proximity_gated` 速度惩罚（二次速度 × gate） |
  | `orientation_penalty` | `body_angle` | `quadratic_penalty`（‑w * angle²） |
  | `landing_contact_success` | `left_support_contact`, `right_support_contact`, `x_velocity`, `y_velocity` | `binary_AND_gate` + `velocity_threshold` → 固定正奖励 |

- **排除的职责及原因**  
  - `fuel_efficiency`（`action`）：v1 阶段优先保证成功着陆，过早惩罚引擎使用会压制探索，留待后续迭代。  
  - `angular_velocity_smoothing`：不是核心约束，v1 先关注姿态角本身。  
  - `time_efficiency_penalty`：无 episode 长度截断，固定步惩罚可能干扰着陆策略。  
  - `hard_success_flag`：环境未提供显式 `info` 成功标志，不可用。  
  - `explicit_episodic_time_limit`：同样不可用。

- **为什么没有 `terminal_success_reward` / `terminal_failure_penalty`**  
  环境卡片明确 `explicit_success_flag_available=false`、`explicit_failure_flag_available=false`，`info` 为空。不能也无依据构造一次性终端奖励。

- **留到后续迭代的职责**  
  - `fuel_efficiency`（加入极小动作代价，但需避免提前压制推力）。  
  - `angular_velocity_smoothing`（若训练中出现姿态振荡则加入）。  
  - `time_efficiency_penalty`（若观察到大量悬停延迟行为，可加入极小的每步惩罚）。  
  - 更复杂的 `curriculum_weighting` 或多阶段门控（如根据训练进度调节速度门限）。

- **训练后应观察的 failure modes**  
  - **高速撞击目标垫**：y_velocity 仍为较大负值便落地，可能被 `soft_landing` gate 捕获不足；需收紧 gate 的 `k_dist` 或增大速度惩罚权重。  
  - **单腿着陆侧翻**：`landing_contact_success` 仅奖励双腿，若 agent 只放一条腿则得不到 bonus；可增加对单腿接触的负向提示（未来迭代）。  
  - **停滞不动**：如果 `progress` 分量始终为零且动作均选 `no_engine`，说明缺乏探索激励；可加入极小动作熵奖励或增大 `w_progress`。  
  - **姿态摆动**：若 `orientation_penalty` 不够强，agent 可能反复倾斜再纠正；检查角度惩罚量级，必要时加入 `angular_velocity` 惩罚。  
  - **引擎浪费（无意义的左/右引擎）**：目前未惩罚，若 agent 频繁闪烁方向引擎但位置无变化，可在下一版加入小权重动作代价。
