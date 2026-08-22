# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract key signals from next_obs
    body_z = next_obs[0]
    quat_w = next_obs[1]   # not used directly but kept for completeness
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]   # not used directly
    
    forward_vel = next_obs[13]
    lateral_vel = next_obs[14]
    vertical_vel = next_obs[15]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    
    # Derived upright metric: body_up_z
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    
    # --- Reward Components ---
    
    # Main progress: forward velocity (dense, linear)
    forward_reward = 5.0 * forward_vel
    
    # Predictive descent penalty: downward velocity when height is losing margin
    descent_penalty = 0.0
    if body_z < 0.5 and vertical_vel < 0.0:
        descent_penalty = -0.5 * abs(vertical_vel) * (0.5 - body_z)
    
    # Predictive ascent penalty: upward velocity when height is too high
    ascent_penalty = 0.0
    if body_z > 0.8 and vertical_vel > 0.0:
        ascent_penalty = -0.3 * vertical_vel * (body_z - 0.8)
    
    # Lateral drift hinge: penalise excessive sideways motion
    lateral_penalty = -0.2 * max(0.0, abs(lateral_vel) - 0.3)
    
    # Roll / pitch angular velocity hinge (avoids rapid tilting)
    roll_vel_penalty = -0.1 * max(0.0, abs(roll_vel) - 3.0)
    pitch_vel_penalty = -0.1 * max(0.0, abs(pitch_vel) - 3.0)
    
    # Upright tilt penalty: punish large body tilt
    upright_penalty = -0.5 * max(0.0, 0.6 - body_up_z)
    
    # Joint velocity regularisation (very light)
    joint_vels = next_obs[19:27]
    joint_vel_penalty = -0.0005 * sum(v**2 for v in joint_vels)
    
    # Action magnitude regularisation (tiny, analogous to energy cost)
    action_penalty = -0.001 * sum(a**2 for a in action)
    
    total = (forward_reward + descent_penalty + ascent_penalty +
             lateral_penalty + roll_vel_penalty + pitch_vel_penalty +
             upright_penalty + joint_vel_penalty + action_penalty)
    
    components = {
        'forward_reward': forward_reward,
        'descent_penalty': descent_penalty,
        'ascent_penalty': ascent_penalty,
        'lateral_penalty': lateral_penalty,
        'roll_vel_penalty': roll_vel_penalty,
        'pitch_vel_penalty': pitch_vel_penalty,
        'upright_penalty': upright_penalty,
        'joint_vel_penalty': joint_vel_penalty,
        'action_penalty': action_penalty
    }
    
    return float(total), components
```

# reward_v1 设计说明

## 任务与环境画像
- **task_family**: 3D 四足机器人持续前进（locomotion_continuous_control, multi_legged_body_locomotion）
- **控制类型**: 连续扭矩控制（8 关节）
- **核心目标**: 维持身体高度在 (0.2, 1.0)、保持直立、产生持续的前进速度；次要目标为最大化前进速度
- **信号可用性**: 有身体高度、四元数、身体速度、关节速度等密集信号；无显式成功/失败标志，无全局位置，无接触信息

## 设计思路：预测性惩罚代替硬门控

历史实验表明，将前进奖励与健康门控（gate）相乘极易导致奖励塌缩或探索不足，单纯的多项独立惩罚又使总奖励过度负化。本次设计采取“**主奖励 + 预测性轻量惩罚**”策略，核心假设是：

- **主奖励（forward_reward）** 提供直接的向前驱动力，使用 `5.0 * forward_vel`（线性），数值足够大以保证正反馈；
- **预测性惩罚** 在危险动作（下降速度、上升速度、侧向漂移、快速翻滚/俯仰角速度）刚刚出现时给予轻度梯度，防止 agent 滑向不可恢复的失败状态，而不是等身体已经越界再施加过强惩罚；
- **轻量约束**（关节速度、动作幅度）抑制高频振荡，但不压制探索。

这种结构从 iter 4 的唯一正分实验（descent_early_penalty + forward_progress + 少量约束）获得启发，并将其扩展为更全面的预测规则。

## 选择的奖励职责与公式算子映射

| 职责 (role) | 使用信号 | 公式算子 | 权重与阈值 | 说明 |
|---|---|---|---|---|
| forward_velocity_progress | next_obs[13] | dense_state_signal (线性) | `+5.0 * forward_vel` | 核心学习信号，推动机器人持续前进 |
| descent_early_penalty | next_obs[0], [15] | joint_condition_proxy (乘积 hinge) | `-0.5 * \|v_z\| * max(0, 0.5 - h)`（当 h<0.5 且 v_z<0） | 提前惩罚下降速度，避免摔倒前兆无反馈 |
| ascent_early_penalty | next_obs[0], [15] | joint_condition_proxy (乘积 hinge) | `-0.3 * v_z * max(0, h - 0.8)`（当 h>0.8 且 v_z>0） | 防止跳起导致高度越界 |
| lateral_drift_penalty | next_obs[14] | dense_state_signal (hinge) | `-0.2 * max(0, \|v_y\| - 0.3)` | 抑制侧向漂移 |
| roll_pitch_angular_penalty | next_obs[16], [17] | dense_state_signal (hinge) | `-0.1 * max(0, \|\omega_roll\| - 3.0)` 等同 | 防止快速翻滚/俯仰，避免失稳 |
| upright_penalty | next_obs[2:3] → body_up_z | dense_state_signal (hinge) | `-0.5 * max(0, 0.6 - body_up_z)` | 倾斜过大时给予温和惩罚 |
| joint_velocity_regularization | next_obs[19:27] | quadratic_penalty (L2) | `-0.0005 * Σ v_joint²` | 极轻量抑制关节振荡 |
| action_magnitude | action[0:7] | quadratic_penalty (L2) | `-0.001 * Σ a²` | 微小的动作幅度成本 |

## 排除的角色及原因

- **healthy_height_gate**（原 mandatory）：多次实验证明门控会导致奖励塌缩；本设计通过预测性惩罚间接实现保护，效果更平滑。
- **upright_penalty 作为独立 gate**：已替换为轻量 hinge 惩罚，避免关闭前进奖励。
- **energy_efficiency**：无直接能量信号，且任务未要求省能。
- **target_reaching / distance_to_goal**：无全局 x 坐标。
- **foot_contact_reward**：
