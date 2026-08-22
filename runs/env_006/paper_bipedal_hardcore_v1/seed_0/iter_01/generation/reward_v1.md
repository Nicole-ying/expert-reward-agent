# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Bipedal locomotion reward for rough terrain:
    - Primary: forward velocity reward with soft health gate based on hull stability
    - Constraint: postural hinge penalty for extreme tilt

    role-based component budget: 2 components (forward_progress + postural_stability)
    """
    # ==================== Extract signals ====================
    # Current hull state
    hull_angle = obs[0]
    hull_angvel = abs(obs[1])

    # Next hull state
    next_hull_angle = next_obs[0]
    next_hull_angvel = abs(next_obs[1])

    # Forward velocity (next step)
    horizontal_speed = next_obs[2]

    # ==================== Constants ====================
    # Hull tilt safety thresholds (radians, empirically near falling boundary)
    TILT_CRITICAL = 0.6      # near-falling severe tilt
    TILT_WARNING_START = 0.25  # begin gentle attenuation
    TILT_WARNING_MARGIN = 0.35  # attenuation window width

    # Hinge penalty thresholds
    HINGE_THRESHOLD = 0.35   # start penalizing tilt above this
    HINGE_SCALE = 1.0

    # Weights (balanced for per-step magnitude comparable)
    FORWARD_WEIGHT = 2.0
    POSTURE_HINGE_WEIGHT = 0.5

    # ==================== Component A: Forward progress with soft health gate ====================
    # Gate factor: linear attenuation from 1.0 (safe) to 0.0 (critical)
    # Uses next_hull_angle (immediate future stability) to gate reward
    abs_tilt = abs(next_hull_angle)
    if abs_tilt <= TILT_WARNING_START:
        gate = 1.0
    elif abs_tilt >= TILT_CRITICAL:
        gate = 0.0
    else:
        gate = (TILT_CRITICAL - abs_tilt) / TILT_WARNING_MARGIN

    # Forward reward: convex to encourage speed, not just minimal forward motion
    forward_reward = FORWARD_WEIGHT * horizontal_speed ** 2
    gated_forward = gate * forward_reward

    # ==================== Component B: Postural hinge penalty ====================
    # Only penalize when tilt exceeds safe threshold
    # Penalizes both current and next tilt, and angular velocity
    current_excess = max(0.0, abs(hull_angle) - HINGE_THRESHOLD)
    next_excess = max(0.0, abs_tilt - HINGE_THRESHOLD)

    # Average excess over the step (current + next) with velocity penalty
    tilt_penalty = HINGE_SCALE * (current_excess + next_excess) * 0.5
    angvel_penalty = 0.3 * next_hull_angvel  # angular velocity contributes to instability

    posture_penalty = -POSTURE_HINGE_WEIGHT * (tilt_penalty + angvel_penalty)

    # ==================== Total reward ====================
    total_reward = gated_forward + posture_penalty

    # ==================== Components dict ====================
    components = {
        'gated_forward_speed': gated_forward,
        'posture_hinge_penalty': posture_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与动态子类型
- **task_family**: locomotion_continuous_control
- **dynamics_subtype**: planar_bipedal_gait on rough terrain
- **control_type**: continuous torque-controlled (hip × 2, knee × 2)

## 2. 选择的奖励职责 (selected roles)
根据 `reward_role_decomposition`，v1 聚焦于 mandatory roles:
- **role_id: forward_progress** (主学习信号)
- **role_id: postural_stability** (安全约束，防止摔倒)

## 3. 职责-信号映射 (role_to_signal_mapping)
| Role | Signal | Index | Formula Operator |
|------|--------|-------|------------------|
| forward_progress | horizontal_speed | next_obs[2] | dense_state_signal (凸化: `speed**2`) + soft_health_gate (linear decay) |
| postural_stability | hull_angle | obs[0], next_obs[0] | dense_state_signal (hinge: `max(0, abs(angle) - threshold)`) |
| postural_stability (supplement) | hull_angular_velocity | next_obs[1] | quadratic_penalty (implicitly via hinge addition) |

## 4. 公式选择详情
- **Component A: gated_forward_speed**
  - 基础形式: `FORWARD_WEIGHT * horizontal_speed ** 2` (dense_state_signal 凸化)
  - 设计意图: 鼓励持续向前移动；平方项使 agent 不会满足于极低速度的平台期
  - 软门控: `gate * forward_reward` (soft_health_gate)
    - 门控形式: linear decay from `TILT_WARNING_START=0.25` to `TILT_CRITICAL=0.6`
    - 设计意图: agent 在身体倾斜接近危险范围时，速度奖励逐渐衰减，引导其优先恢复姿态而不是继续冲撞
    - 下限为零（gate=0），但仅在极端倾斜时触发

- **Component B: posture_hinge_penalty**
  - 基础形式: `-POSTURE_HINGE_WEIGHT * (max(0, abs(angle) - HINGE_THRESHOLD))` (dense_state_signal hinge)
  - 设计意图: 只在倾斜超过安全阈值时施加惩罚，避免在正常行走波动中小幅度姿态变化也被惩罚（全时二次惩罚会抑制探索）
  - 阈值设在 0.35 rad，约在终止临界 (0.5-0.6 rad) 的 60-70%，给予 early warning
  - 额外加入角速度惩罚 (`0.3 * next_hull_angvel`)，抑制快速旋转（急剧翻滚）

## 5. 排除的职责 (excluded roles)
| Role | Reason for Exclusion |
|------|---------------------|
| energy_efficiency | v1 优先学习行走目标；能耗优化留到后续迭代，避免过强动作惩罚抑制步态形成 |
| successful_termination_bonus | explicit_success_flag_available=false; 通过观测推断 reach_end_of_terrain 不可靠，容易将中间静止误认为成功 |
| terminal_failure_penalty | avoid_roles 已说明：无 direct_failure_flag；用 hull_angle 阈值等价于 postural_stability 加倍，不必重复 |
| lidar_terrain_anticipation | LIDAR 信号无法直接转化为单标量奖赏；让 RL 自主学习 LIDAR-动作映射，v1 不显式建模为奖励项 |

## 6. 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- environment_card 明确 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`
- info 为空，无法读取任何终止标记
- 从观测推断终态的可靠性不足，尤其 success (reached_end_of_terrain) 容易误判
- v1 将终态信号坍缩进稠密的 postural_stability 和 soft_health_gate，无需独立的稀疏终端项

## 7. 留到后续迭代的职责
- **energy_efficiency**: 可加入轻量 `-w * sum(action_i**2)`（参考 action_efficiency 算子）
- **lidar_terrain_anticipation**: 如果后续观测到 agent 频繁在障碍物前摔倒，可设计 preview_conditioned_reward 或 terrain-aware gate
- **joint_condition_proxy**: 如果添加 success proxy，需要多条件组合（hull tilt + speed + contact），当前 v1 无此需求

## 8. 训练后应观察的 failure modes
1. **velocity_burst_then_fall**: 速度奖励平方项可能诱导短期高速冲刺 → 如果结合 soft_health_gate 效果不足，可能需提升门控衰减斜率或降低 FORWARD_WEIGHT
2. **stand_still**: 惩罚过强（posture_hinge + angvel）可能导致 agent 不敢迈步 → 可降低 POSTURE_HINGE_WEIGHT 或提高 HINGE_THRESHOLD
3. **foot_contact_hacking**: 当前未使用 ground_contact 信号，agent 可能学习到单腿跳跃前进的捷径 → 后续可加入双支撑相的最小比例约束
4. **gate_exploitation**: agent 可能学习到刻意保持 gate=0 来避免惩罚（但这样速度奖励也为 0）→ 需观察 episode 内 gate 的分布，确认是否出现消极策略
5. **terrain_ignorance**: 未使用 LIDAR，agent 可能在障碍物前失败 → 如果 performance 停滞，后续可引入 preview_conditioned_reward