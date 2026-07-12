# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号
    #    使用 horizontal_velocity (obs[2]) 作为核心驱动力
    #    采用凸化形式 (signal**2) 以打破低水平稳态
    # ============================================================
    horizontal_vel = obs[2]
    # 凸化正奖励：鼓励持续前进，避免满足于低速
    forward_progress_reward = 1.0 * (horizontal_vel ** 2)
    
    # ============================================================
    # 2. balance_maintenance: 健康约束
    #    使用 hull_angle (obs[0]) 和 hull_angular_velocity (obs[1])
    #    采用 hinge 形式：只在角度偏离超过阈值时惩罚
    #    阈值设为 0.3 rad (~17度)，在终止边界前提供软约束
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    
    # 角度偏离惩罚：只在 |angle| > 0.3 时生效
    angle_threshold = 0.3
    angle_penalty = -0.5 * max(0.0, abs(hull_angle) - angle_threshold) ** 2
    
    # 角速度惩罚：只在角速度过大时生效
    angular_vel_threshold = 2.0
    angular_vel_penalty = -0.1 * max(0.0, abs(hull_angular_vel) - angular_vel_threshold) ** 2
    
    balance_penalty = angle_penalty + angular_vel_penalty
    
    # ============================================================
    # 3. soft_health_gate: 将 balance 状态作为 gate 乘到主奖励上
    #    当 hull_angle 接近危险值时，主奖励平滑衰减
    #    这比单纯加惩罚更有效，因为 agent 在恶化前就会失去前进动力
    # ============================================================
    # 门控因子：角度偏离越大，门控越小
    # 在 0.0~0.5 rad 范围内线性衰减
    gate_factor = max(0.0, min(1.0, 1.0 - abs(hull_angle) / 0.5))
    
    # 应用门控到主奖励
    gated_forward_reward = forward_progress_reward * gate_factor
    
    # ============================================================
    # 4. 组合总奖励
    # ============================================================
    total_reward = gated_forward_reward + balance_penalty
    
    # ============================================================
    # 组件记录
    # ============================================================
    components = {
        'forward_progress_reward': forward_progress_reward,
        'gated_forward_reward': gated_forward_reward,
        'balance_angle_penalty': angle_penalty,
        'balance_angular_vel_penalty': angular_vel_penalty,
        'gate_factor': gate_factor
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务类型与动力学子类型
- **task_family**: locomotion_continuous_control
- **dynamics_subtype**: planar_bipedal_gait

## 选中的奖励职责 (selected reward roles)
1. **forward_progress** (主学习信号) — 必须包含
2. **balance_maintenance** (健康约束) — 必须包含

## role_to_signal_mapping
| role_id | usable signals | formula operator | 具体实现 |
|---|---|---|---|
| forward_progress | horizontal_velocity (obs[2]) | dense_state_signal (凸化) | `1.0 * vel^2` |
| balance_maintenance | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | dense_state_signal (hinge) | `-0.5 * max(0, \|angle\|-0.3)^2` 和 `-0.1 * max(0, \|ang_vel\|-2.0)^2` |

## 额外使用的算子
- **soft_health_gate**: 将 balance 状态作为门控乘到主奖励上，在角度偏离超过 0.5 rad 时完全切断主奖励。这比单纯加惩罚更有效，因为 agent 在恶化前就会失去前进动力。

## 排除的职责 (excluded roles) 及原因
| excluded role | 原因 |
|---|---|
| energy_efficiency | v1 阶段避免过早加入能耗约束，防止 agent 不敢动作 |
| gait_smoothness | v1 阶段优先学习前进和平衡，步态优化留到后续迭代 |
| terrain_adaptation | LIDAR 信号噪声大，直接使用可能导致不稳定，留到后续迭代 |
| contact_rhythm | 缺少步态相位信号，接触标志只有二值，无法构建稳定节奏奖励 |
| height_maintenance | 缺少垂直位置观测，无法实现 |
| goal_reaching | 任务不是到达特定目标点，而是持续前进 |

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- **explicit_success_flag_available**: false — info 为空字典，没有显式 success 标志
- **explicit_failure_flag_available**: false — info 为空字典，没有显式 failure 标志
- 因此无法使用基于终止条件的稀疏奖励，所有信号必须来自 obs 和 action

## 留到后续迭代的职责
1. **energy_efficiency**: 当基础前进和平衡稳定后，加入动作平方和惩罚
2. **gait_smoothness**: 当步态出现明显抖动时，加入关节角加速度惩罚
3. **terrain_adaptation**: 当 LIDAR 显示地形变化时，加入地形感知奖励
4. **contact_rhythm**: 如果未来有步态相位信号，可加入对称性奖励

## 训练后应观察的 failure modes
| failure_mode | 证据 | 干预方案 |
|---|---|---|
| 过早摔倒 | episode 长度很短，hull_angle 快速偏离 | 增加 balance_penalty 权重，或降低 gate 阈值 |
| 原地踏步不前 | horizontal_velocity 接近 0，但 episode 未终止 | 增加 forward_progress_reward 权重，或改用线性形式 |
| 过度前倾冲刺 | horizontal_velocity 高但 hull_angle 持续偏离 | 增加 angle_penalty 权重，或加入角度-速度耦合惩罚 |
| 步态僵硬 | 关节角度变化小，动作幅度小 | 减少 balance_penalty 权重，或加入 gait_smoothness |
| 动作过大 | action 值接近边界 | 后续迭代加入 energy_efficiency |