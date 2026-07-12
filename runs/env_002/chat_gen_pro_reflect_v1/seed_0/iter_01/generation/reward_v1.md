# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号 - 鼓励向前行走
    #    signal: horizontal_velocity (obs[2])
    #    operator: dense_state_signal (线性正奖励)
    #    使用线性形式而非凸化，因为v1阶段需要鼓励agent探索不同速度
    # ============================================================
    forward_velocity = obs[2]
    forward_reward = 1.0 * forward_velocity  # 系数1.0，使速度值直接作为奖励
    
    # ============================================================
    # 2. balance_maintenance: 健康约束 - 保持身体直立
    #    signals: hull_angle (obs[0]), hull_angular_velocity (obs[1])
    #    operator: dense_state_signal (hinge penalty)
    #    使用hinge形式：只在角度超过安全阈值时惩罚，避免过度约束
    #    安全阈值设为0.5弧度(~28度)，超过后线性增加惩罚
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    
    # 角度惩罚：只在超过0.5弧度时生效
    angle_threshold = 0.5
    angle_penalty = -2.0 * max(0.0, abs(hull_angle) - angle_threshold)
    
    # 角速度阻尼：轻微惩罚角速度，防止快速旋转
    angular_vel_penalty = -0.1 * hull_angular_vel ** 2
    
    balance_penalty = angle_penalty + angular_vel_penalty
    
    # ============================================================
    # 3. soft_health_gate: 将平衡状态作为门控乘到前进奖励上
    #    当身体角度接近危险时，前进奖励逐渐衰减
    #    使用倒数门形式：1 / (1 + k * abs(angle))
    #    这样在角度较小时门控接近1，角度增大时平滑衰减
    # ============================================================
    gate_k = 3.0  # 控制衰减速度
    health_gate = 1.0 / (1.0 + gate_k * abs(hull_angle))
    
    # 应用门控：前进奖励乘以健康门控
    gated_forward_reward = forward_reward * health_gate
    
    # ============================================================
    # 4. 组合最终奖励
    #    total_reward = 门控前进奖励 + 平衡惩罚
    #    门控前进奖励是主要驱动力，平衡惩罚是辅助约束
    # ============================================================
    total_reward = gated_forward_reward + balance_penalty
    
    # ============================================================
    # 组件记录
    # ============================================================
    components = {
        'forward_reward': forward_reward,          # 原始前进奖励（未门控）
        'health_gate': health_gate,                # 健康门控值
        'gated_forward_reward': gated_forward_reward,  # 门控后的前进奖励
        'angle_penalty': angle_penalty,            # 角度越界惩罚
        'angular_vel_penalty': angular_vel_penalty, # 角速度阻尼
        'balance_penalty': balance_penalty         # 总平衡惩罚
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与角色选择

- **task_family**: locomotion_continuous_control
- **dynamics_subtype**: planar_bipedal_gait
- **selected reward roles**: forward_progress (mandatory), balance_maintenance (mandatory)
- **excluded roles**: energy_efficiency, gait_smoothness, terrain_adaptation, contact_rhythm, height_maintenance, distance_traveled

## role_to_signal_mapping

| role_id | usable signals | formula operator | 设计理由 |
|---|---|---|---|
| forward_progress | horizontal_velocity (obs[2]) | dense_state_signal (线性正奖励) | 直接使用水平速度作为正向奖励，线性形式鼓励探索不同速度 |
| balance_maintenance | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | dense_state_signal (hinge penalty) + soft_health_gate | 角度使用hinge惩罚只在越界时生效；角速度使用二次阻尼；同时将角度作为门控乘到前进奖励上 |

## 核心设计决策

### 1. 为什么使用 soft_health_gate 而非独立惩罚
根据 expert_reward_context.md 的触发证据：当 terminated 率高且主进展信号在失败回合仍为正时，应使用 gate 而非加大独立惩罚。本任务中：
- 摔倒直接终止任务
- agent 可能在摔倒前获得大量前进奖励（"先冲后死"模式）
- 使用 health_gate 在角度增大时平滑衰减前进奖励，比单纯惩罚更有效

### 2. 为什么使用 hinge 形式的角度惩罚
- 全时二次惩罚（`-w * angle^2`）会让 agent 不敢偏离直立状态，抑制探索
- hinge 形式只在角度超过 0.5 弧度时惩罚，给 agent 安全探索空间
- 0.5 弧度（~28度）是合理的"安全边界"，超过后需要纠正

### 3. 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- explicit_success_flag_available = false（info为空字典）
- explicit_failure_flag_available = false（info为空字典）
- 无法在终止时区分成功或失败，因此不使用终端奖励

### 4. 为什么没有使用 energy_efficiency
- v1 阶段优先让 agent 学会前进和平衡
- 过早加入能耗惩罚可能阻碍探索有效步态
- 留到后续迭代（v2/v3）加入

### 5. 为什么没有使用 gait_smoothness
- 需要关节角速度信号（obs[5,7,10,12]），信号可用
- 但 v1 阶段应避免过度约束关节运动
- 步态平滑性优化留到后续迭代

## 训练后应观察的 failure modes

| failure_mode | 观察证据 | 可能干预 |
|---|---|---|
| 身体前倾摔倒 | hull_angle 持续增大，forward_velocity 突然下降 | 降低 angle_threshold，增加 angle_penalty 系数，或增大 gate_k |
| 原地跳跃不前进 | forward_velocity 接近0，vertical_velocity 波动大 | 增加 forward_reward 系数，或加入 vertical_velocity 惩罚 |
| 动作幅度过大 | action 值接近 ±1 | 后续迭代加入 energy_efficiency 惩罚 |
| 过早终止 | 平均 episode 长度短，摔倒率高 | 检查 angle_threshold 是否太严格，或 gate_k 是否过大 |
| 慢速行走 | forward_velocity 持续偏低但稳定 | 将 forward_reward 改为凸化形式（`velocity^2`） |