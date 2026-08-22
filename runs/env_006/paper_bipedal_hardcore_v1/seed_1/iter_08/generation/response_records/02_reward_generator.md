# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测信号（所有索引均基于 environment_card 声明的 24 维观测）
    hull_angle = obs[0]
    hull_angvel = obs[1]
    horizontal_speed = obs[2]
    vertical_speed = obs[3]

    # ========== 组件 A：前进主奖励 ==========
    # 直接以水平速度作为稠密进展信号，惩罚向后运动
    w_forward = 1.0
    forward_progress = w_forward * horizontal_speed

    # ========== 组件 B：身体倾角稳定性惩罚 ==========
    # 当倾角超过安全阈值（0.5 rad ≈ 28.6°）时施加二次惩罚，越界越多惩罚急剧增大
    angle_threshold = 0.5
    w_angle = 5.0
    angle_error = max(0.0, abs(hull_angle) - angle_threshold)
    stability_angle_penalty = -w_angle * (angle_error ** 2)

    # ========== 组件 C：身体角速度稳定性惩罚 ==========
    # 角速度超过阈值时二次惩罚，抑制急转与即将摔倒的快速旋转
    angvel_threshold = 1.5
    w_angvel = 0.5
    angvel_error = max(0.0, abs(hull_angvel) - angvel_threshold)
    stability_angvel_penalty = -w_angvel * (angvel_error ** 2)

    # ========== 组件 D：垂直速度骤降惩罚 ==========
    # 当机器人向下坠落速度过快时惩罚，预防硬着陆或摔倒
    v_threshold = 0.5
    w_v = 2.0
    v_error = max(0.0, -vertical_speed - v_threshold)  # 只有向下且超过阈值才考虑
    vertical_speed_penalty = -w_v * (v_error ** 2)

    # 总奖励为所有组件的和
    total_reward = forward_progress + stability_angle_penalty + stability_angvel_penalty + vertical_speed_penalty

    components = {
        "forward_progress": forward_progress,
        "stability_angle_penalty": stability_angle_penalty,
        "stability_angvel_penalty": stability_angvel_penalty,
        "vertical_speed_penalty": vertical_speed_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像匹配
- **task_family**: locomotion_continuous_control (planar_bipedal_gait_on_rough_terrain)
- **dynamics_subtype**: 双足扭矩驱动机器人在粗糙地形上行走，LIDAR 提供前方测距。

## 选定的奖励职责与信号映射
| 职责 | 角色 | 信号 | 公式算子 | 设计意图 |
|---|---|---|---|---|
| forward_progress (主学习信号) | 核心进步 | `horizontal_speed` (obs[2]) | dense_state_signal (线性) | 直接驱动向前运动，每步提供稠密梯度；负速度自动产生负奖励从而抑制后退。 |
| survival_balance (平衡约束) | 安全约束 | `hull_angle` (obs[0]) | hinge + quadratic | 只在倾角超过安全阈值（0.5 rad）时施加二次惩罚，越界越大惩罚急剧上升，迫使快速纠正姿态。 |
| survival_balance (角速度约束) | 安全约束 | `hull_angular_velocity` (obs[1]) | hinge + quadratic | 抑制异常的快速旋转，阈值设为1.5 rad/s，避免摔倒前的剧烈摆动。 |
| survival_balance (坠落约束) | 安全约束 | `vertical_speed` (obs[3]) | hinge + quadratic | 惩罚剧烈向下运动（超过0.5 m/s），防止硬着陆或坠落式摔倒。 |

## 排除的职责及原因
- **efficient_actuation**: 历史试验表明在初始阶段加入动作代价（L2 惩罚）经常导致负分且停滞；v1 优先让机器人学会安全前进，暂不加入效率约束。
- **gait_smoothness**: 关节平滑度可能与崎岖地形适应冲突，留待后续迭代。
- **preview_conditioned_reward (LIDAR)**: 地形适应建模风险高，容易导致策略利用传感器而非真正学会迈步，v1 不使用。
- **terminal_success_reward / terminal_failure_penalty**: 环境中没有显式成功/失败标志，info 为空；避免推断带来的不确定性，改为通过连续惩罚函数（二次形状）提供类似“终端惩罚”的强梯度。

## 为什么没有使用 terminal 硬信号
- 环境未提供 `info["success"]` 或 termination reason，任何硬推断（如根据倾角判断摔倒）可能在到达终点时误触发，引入不稳定因素。
- 采用 shape 尖锐的二次惩罚（对倾角、角速度、垂直速度）可以在危险区间提供强梯度同时保留连续优化路径，比单次硬覆盖更利于策略学习。

## 留到后续迭代的职责
- **efficient_actuation**: 当 agent 稳定前进后，逐步加入小权重扭矩惩罚以优化能耗。
- **gait_smoothness**: 可在基本步态形成后作为微小 bonus 加入，使步行更平稳。
- **lidar_aware_preview**: 若能找到稳健的“前景衰减”因子（如前方高度方差过大的门控），可以提前调整步态，改善通过障碍的能力。

## 设计原则自检
1. **终止条件前兆软信号**：倾角持续增大、角速度飙升、垂直速度骤降均为摔倒的前兆，通过惩罚梯度提供早期纠正信号。
2. **任务目标进度信号**：`forward_progress` 是直接的每步速度驱动，保证任务核心方向。
3. **动作维度 ≥ 6 时效率约束**：本环境动作维度 = 4，不大于 6 阈值，且效率惩罚历史效果差，故暂不加。
4. **奖励尺度平衡**：正常行走时惩罚项全部为 0，主要由 `forward_progress`（~1-2/step）驱动；姿态危险时惩罚项迅速增大（倾角0.8 rad可产生-1.25），能有效遏制危险行为而不淹没主要目标。

## 训练后应观察的 failure modes
- 机器人仍频繁摔倒（episode length 短）：若倾角惩罚触发率超高，需要降低角度、角速度阈值或权重；若触发率低但仍摔倒，可能需要增加垂直速度或接触一致性约束。
- 机器人学会缓慢移动以避免惩罚：步进速度显著低于环境所需的合理值，可适当降低稳定性惩罚权重或提高前进权重。
- 对障碍物无法应对、经常停滞：下一步可考虑加入前方 LIDAR 信息构造预判奖励，但需避免策略利用问题。
