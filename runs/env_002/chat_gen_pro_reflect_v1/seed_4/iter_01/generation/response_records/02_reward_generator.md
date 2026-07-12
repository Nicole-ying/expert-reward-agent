# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    # forward_progress: horizontal_velocity (obs[2])
    horizontal_vel = obs[2]
    
    # balance_maintenance: hull_angle (obs[0]), hull_angular_velocity (obs[1])
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    
    # ============================================================
    # Component 1: forward_progress (main learning signal)
    # Use dense_state_signal: positive linear reward for forward velocity
    # Scale: typical horizontal velocity range ~0-5 m/s, target ~2-3 m/s
    # Weight chosen so that at 2 m/s, this component ≈ 2.0
    # ============================================================
    forward_reward = 1.0 * horizontal_vel
    
    # ============================================================
    # Component 2: balance_maintenance (stability constraint)
    # Use dense_state_signal with quadratic penalty for hull_angle deviation
    # hull_angle is in radians, typical stable range ~[-0.3, 0.3]
    # Penalty is mild: at 0.3 rad (~17 deg), penalty ≈ -0.09
    # Also add small penalty for angular velocity to discourage wobbling
    # ============================================================
    angle_penalty = -2.0 * (hull_angle ** 2)
    angular_vel_penalty = -0.1 * (hull_angular_vel ** 2)
    balance_penalty = angle_penalty + angular_vel_penalty
    
    # ============================================================
    # Component 3: soft_health_gate on forward_reward
    # When hull_angle approaches dangerous territory (>0.5 rad), 
    # reduce forward_reward to discourage "rush then fall" behavior
    # Use linear decay gate: gate = max(0, 1 - |hull_angle| / 0.6)
    # At hull_angle=0.3, gate ≈ 0.5; at hull_angle=0.6, gate ≈ 0.0
    # ============================================================
    gate_factor = max(0.0, 1.0 - abs(hull_angle) / 0.6)
    gated_forward = forward_reward * gate_factor
    
    # ============================================================
    # Total reward
    # ============================================================
    total_reward = gated_forward + balance_penalty
    
    # ============================================================
    # Components dict (for debugging and monitoring)
    # ============================================================
    components = {
        'forward_reward': forward_reward,
        'gated_forward': gated_forward,
        'gate_factor': gate_factor,
        'angle_penalty': angle_penalty,
        'angular_vel_penalty': angular_vel_penalty,
        'balance_penalty': balance_penalty
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务类型与职责选择

- **task_family**: locomotion_continuous_control
- **dynamics_subtype**: planar_bipedal_gait
- **selected reward roles**: `forward_progress` (mandatory), `balance_maintenance` (mandatory)
- **excluded roles**: `energy_efficiency`, `terrain_adaptation`, `gait_smoothness`, `contact_phase_control`, `absolute_position_reaching`, `terrain_height_avoidance`

## 2. role_to_signal_mapping

| role_id | usable signals | formula operator | implementation |
|---|---|---|---|
| forward_progress | horizontal_velocity (obs[2]) | dense_state_signal (linear positive) | `1.0 * horizontal_vel` |
| balance_maintenance | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | dense_state_signal (quadratic penalty) | `-2.0 * hull_angle**2 - 0.1 * hull_angular_vel**2` |

## 3. 核心设计决策

### 为什么使用 soft_health_gate 而非独立惩罚？
根据 environment_card.md 的 failure mode 分析，最可能的失败模式是"身体前倾摔倒"——agent 为了追求速度而过度前倾，最终摔倒。如果只加独立的角度惩罚，agent 可能在摔倒前仍然获得大量前进奖励（"先冲后死"模式）。使用 soft_health_gate 可以在角度恶化时平滑削减前进奖励，让 agent 学会在保持平衡的前提下前进，而不是在摔倒前冲刺。

### 为什么排除 energy_efficiency？
v1 阶段优先让 agent 学会前进和保持平衡。过早加入能耗惩罚可能导致 agent 步态无力，无法前进。能耗优化留到后续迭代。

### 为什么排除 terrain_adaptation？
LIDAR 信号 (obs[14..23]) 虽然可用，但地形适应需要更复杂的多条件组合信号。v1 阶段先让 agent 学会在平坦地形上稳定行走，后续再引入地形适应。

### 为什么排除 gait_smoothness？
步态平滑性需要动作变化率（temporal difference），但 v1 阶段 agent 可能还没有稳定的步态周期。过早加入平滑惩罚可能抑制探索。

### 为什么没有使用 terminal_success_reward / terminal_failure_penalty？
environment_card.md 明确声明 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`，info 字典为空。因此不能使用任何基于 success/failure flag 的终端奖励。

## 4. 训练后应观察的 failure modes

| failure_mode | evidence | intervention |
|---|---|---|
| 身体前倾摔倒 | hull_angle 持续增大，horizontal_velocity 突然归零 | 增加 angle_penalty 权重，或降低 gate_factor 的衰减阈值 |
| 步态僵硬无法前进 | horizontal_velocity 接近零，action 幅度小 | 降低 balance_penalty 权重，或增加 forward_reward 系数 |
| 步态抖动不稳定 | hull_angular_velocity 波动大 | 增加 angular_vel_penalty 权重，或引入 gait_smoothness |
| 能量消耗过高 | action 幅度大但前进速度低 | 后续迭代加入 energy_efficiency 职责 |
