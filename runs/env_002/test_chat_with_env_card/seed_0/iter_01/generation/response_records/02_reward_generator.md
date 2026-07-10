# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. forward_progress: 主学习信号
    #    使用 horizontal_velocity (obs[2]) 作为核心驱动力
    #    采用线性正奖励，鼓励向前行走
    # ============================================================
    horizontal_vel = obs[2]
    forward_progress_reward = 1.0 * horizontal_vel  # 权重1.0，线性正奖励

    # ============================================================
    # 2. balance_maintenance: 稳定/安全约束
    #    使用 hull_angle (obs[0]) 和 hull_angular_velocity (obs[1])
    #    采用 quadratic_penalty 形式，惩罚身体倾斜和角速度
    # ============================================================
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    
    # 惩罚身体倾斜：角度偏离竖直方向（0为竖直）
    angle_penalty = -0.5 * (hull_angle ** 2)
    # 惩罚角速度过大
    angular_vel_penalty = -0.1 * (hull_angular_vel ** 2)
    
    balance_penalty = angle_penalty + angular_vel_penalty

    # ============================================================
    # 3. 组合总奖励
    #    v1阶段：主学习信号 + 必要平衡约束
    #    不加入 energy_efficiency（留到后续迭代）
    #    不加入 gait_rhythm（留到后续迭代）
    #    不加入 terrain_adaptation（留到后续迭代）
    # ============================================================
    total_reward = forward_progress_reward + balance_penalty

    # ============================================================
    # 4. 组件记录
    # ============================================================
    components = {
        "forward_progress_reward": forward_progress_reward,
        "balance_penalty": balance_penalty,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务类型与动力学子类型
- **task_family**: locomotion_continuous_control
- **dynamics_subtype**: planar_bipedal_gait
- 核心目标：在崎岖地形上持续向前行走，保持平衡不摔倒

## 2. 选中的奖励职责 (selected reward roles)
| role_id | 角色类型 | 选择理由 |
|---|---|---|
| forward_progress | mandatory | 核心任务目标，必须提供正向驱动力 |
| balance_maintenance | mandatory | 摔倒即终止，必须约束身体姿态 |

## 3. role_to_signal_mapping
| role_id | 可用信号 | 选择的公式算子 | 数学形式 |
|---|---|---|---|
| forward_progress | horizontal_velocity (obs[2]) | dense_state_signal (线性正) | `1.0 * obs[2]` |
| balance_maintenance | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | quadratic_penalty | `-0.5 * obs[0]^2 - 0.1 * obs[1]^2` |

## 4. 公式算子选择理由
- **forward_progress**: 使用线性正奖励而非凸化形式。v1阶段agent需要先学会"向前走"的基本概念，线性形式提供稳定梯度，避免过早凸化导致agent追求极端速度而摔倒。
- **balance_maintenance**: 使用quadratic_penalty而非hinge形式。quadratic从中心（竖直状态）开始平滑惩罚，鼓励agent保持直立；hinge只在越界时惩罚，可能让agent在安全范围内自由倾斜，不利于早期步态学习。

## 5. 排除的职责及原因
| 排除的role_id | 排除原因 |
|---|---|
| energy_efficiency | v1阶段agent应先学会行走，过早加入能耗惩罚可能导致不敢动作 |
| gait_rhythm | 需要步态相位推断，v1阶段信号处理复杂，留到后续迭代 |
| terrain_adaptation | LIDAR信号维度高(10维)，v1阶段增加学习难度 |
| speed_maximization | 过度追求速度可能导致步态不稳定，与balance_maintenance冲突 |
| joint_angle_limits | 环境物理引擎可能已内置关节限位，且无关节限位观测信号 |
| contact_force_minimization | 无接触力观测信号 |

## 6. 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- **explicit_success_flag_available**: false（info字典为空，无success字段）
- **explicit_failure_flag_available**: false（info字典为空，无failure字段）
- 因此无法使用终端奖励，所有奖励必须基于每步的连续信号

## 7. 留到后续迭代的职责
| 职责 | 计划迭代阶段 | 预期信号/方法 |
|---|---|---|
| energy_efficiency | v2 | action平方和惩罚，权重从0.01开始 |
| gait_rhythm | v2 | 基于leg_contact的交替步态奖励 |
| terrain_adaptation | v3 | LIDAR信号与步态调整的联合建模 |
| soft_health_gate | v2 | 如果terminated率高，将balance作为gate乘到forward_progress上 |

## 8. 训练后应观察的 failure modes
| failure_mode | 观察指标 | 可能干预 |
|---|---|---|
| 摔倒频繁 | terminated率>50%，hull_angle绝对值>0.5 | 增加balance_penalty权重，或改用soft_health_gate |
| 原地不动/后退 | horizontal_velocity均值<0.1 | 增加forward_progress权重，或改用凸化形式 |
| 步态不对称 | 一条腿接触时间远多于另一条 | v2加入gait_rhythm |
| 动作抖动/高频振荡 | action变化率大 | v2加入energy_efficiency |
| 地形适应不良 | 在LIDAR显示障碍时摔倒 | v3加入terrain_adaptation |
