# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant observations
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]
    horizontal_velocity = obs[2]
    vertical_velocity = obs[3]
    leg1_contact = obs[8]
    leg2_contact = obs[13]
    
    # ========== Component 1: Forward velocity reward (main learning signal) ==========
    # Encourage fast forward movement. Use a linear reward capped at a reasonable speed.
    # This provides a strong, continuous gradient for the primary task.
    target_speed = 2.0  # m/s, reasonable walking speed for a biped
    forward_reward = min(horizontal_velocity / target_speed, 1.0)  # normalized to [0, 1]
    
    # ========== Component 2: Upright posture penalty (stability constraint) ==========
    # Penalize deviation from upright posture. Use a quadratic penalty to strongly discourage
    # falling over while allowing small natural sway during walking.
    max_allowed_angle = 0.5  # radians (~28 degrees)
    angle_penalty = (hull_angle / max_allowed_angle) ** 2  # quadratic penalty
    
    # ========== Component 3: Ground contact bonus (gait quality) ==========
    # Reward alternating ground contact to encourage proper gait cycle.
    # Having at least one foot on ground is essential for stability.
    contact_bonus = 0.5 * (leg1_contact + leg2_contact)  # [0, 1], higher when both feet contact
    
    # ========== Component 4: Energy penalty (efficiency constraint) ==========
    # Penalize large joint torques to discourage wasteful flailing.
    # Use squared action norm for smoothness and energy efficiency.
    energy_penalty = 0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)
    
    # ========== Combine components ==========
    # Weights chosen to balance the components:
    # - forward_reward: primary signal, weight 2.0
    # - angle_penalty: critical for survival, weight 1.0 (subtracted)
    # - contact_bonus: auxiliary signal, weight 0.5
    # - energy_penalty: small penalty, weight 0.01 (already scaled)
    total_reward = 2.0 * forward_reward - 1.0 * angle_penalty + 0.5 * contact_bonus - energy_penalty
    
    components = {
        'forward_reward': forward_reward,
        'angle_penalty': angle_penalty,
        'contact_bonus': contact_bonus,
        'energy_penalty': energy_penalty
    }
    
    return float(total_reward), components
```

# 设计说明

**任务类型判断**：2D bipedal locomotion with uneven terrain. The agent must walk forward while maintaining balance.

**信号选择与职责**：
1. **Forward velocity reward** (主学习信号)：直接使用 `horizontal_velocity` (obs[2])，这是任务的核心目标。线性归一化到 [0,1] 提供平滑梯度，避免 agent 追求无限速度导致不稳定。
2. **Upright posture penalty** (健康约束)：使用 `hull_angle` (obs[0]) 的二次惩罚。二次形式对小角度偏差容忍度高，对大角度偏差惩罚急剧增加，有效防止摔倒而不干扰正常步态摆动。
3. **Ground contact bonus** (步态质量辅助)：使用 `leg1_contact` 和 `leg2_contact` (obs[8], obs[13])。鼓励双脚交替着地，促进稳定步态周期。二值信号但通过求和变成连续值 [0,1]。
4. **Energy penalty** (效率约束)：使用动作的平方和，权重 0.01 使其不会主导主信号。防止 agent 通过剧烈摆动关节来获得速度。

**数学形式选择理由**：
- 线性 forward_reward：简单直接，提供恒定梯度
- 二次 angle_penalty：对危险状态指数级惩罚，对安全状态宽容
- 线性 contact_bonus：简单求和，避免复杂门控逻辑
- 二次 energy_penalty：L2 正则化形式，鼓励平滑动作

**排除的职责**：
- 未使用 LIDAR 数据 (obs[14..23])：v1 聚焦基本 locomotion，地形感知留给后续迭代
- 未使用 success/failure 终端奖励：v1 使用稠密信号，终端奖励在早期训练中样本效率低
- 未使用 hull_angular_velocity：角度惩罚已足够，加入角速度可能过度约束

**应观察的 failure modes**：
- **Velocity burst then fall**：如果 agent 学会快速前冲但摔倒，angle_penalty 会惩罚
- **Stand still**：如果 agent 静止不动，forward_reward 为 0，无法获得正向奖励
- **Flailing**：如果 agent 剧烈摆动关节，energy_penalty 会抑制
- **Hopping on one leg**：contact_bonus 鼓励双脚着地，但权重较低，可能需要后续加强