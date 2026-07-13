# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- extract current and next state ----
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # Component A: distance progress (main learning signal)
    prev_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    progress = prev_dist - next_dist
    w_progress = 1.0
    c_progress = w_progress * progress

    # Component B: soft landing bonus (task-completion proxy)
    dist_land = (nx**2 + ny**2) ** 0.5
    speed_land = (nvx**2 + nvy**2) ** 0.5
    contact_flag = n_left_contact * n_right_contact   # 1.0 if both legs touch

    prox_score = 2.718281828 ** (-dist_land**2 / 0.1)
    vel_score  = 2.718281828 ** (-(nvx**2 + nvy**2) / 0.1)
    angle_score = 2.718281828 ** (-(n_angle**2) / 0.01)
    angvel_score = 2.718281828 ** (-(n_angvel**2) / 0.01)

    landing_bonus = contact_flag * prox_score * vel_score * angle_score * angvel_score
    w_landing = 10.0
    c_landing = w_landing * landing_bonus

    # Component C: posture penalty (stability constraint)
    w_angle_pen = 0.1
    w_angvel_pen = 0.001
    c_posture = -w_angle_pen * (n_angle**2) - w_angvel_pen * (n_angvel**2)

    total_reward = c_progress + c_landing + c_posture
    components = {
        'distance_progress': c_progress,
        'soft_landing_bonus': c_landing,
        'posture_penalty': c_posture
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与动力学亚型
- **task_family:** `navigation_goal_reaching`（飞行器导航至中心目标垫并稳定降落）
- **dynamics_subtype:** 2D 飞行器，离散推力（无动作幅度，只有引擎启用/关闭）

## 选择的奖励角色及映射
| 角色                         | 信号来源                               | 公式算子                | 权重/尺度        |
|------------------------------|----------------------------------------|-------------------------|------------------|
| **distance_progress** (主学习信号) | `x_position`, `y_position` (obs与next_obs) | `improvement_delta`     | `w_progress=1.0` |
| **soft_landing_bonus** (任务完成近似) | `left_support_contact`, `right_support_contact`, 位置/速度/姿态/角速度(next_obs) | `joint_condition_proxy` (指数乘积) | `w_landing=10.0` |
| **posture_penalty** (稳定约束) | `body_angle`, `angular_velocity` (next_obs) | `quadratic_penalty`     | `w_angle=0.1`, `w_angvel=0.001` |

- **distance_progress**：每步距离减少量直接驱动 agent 向目标运动，稠密且与核心目标一致。
- **soft_landing_bonus**：利用双腿接触信号作为开关，配合位置接近、低速度、小角度和低角速度的连续指数评分，构造无显式 success flag 的软完成奖励；只在接触发生时给予，引导平稳着陆。
- **posture_penalty**：轻量二次惩罚抑制剧烈姿态变化，防止翻倒或不稳，但不压制正常飞行机动。

## 排除的角色及原因
- **terminal_success_reward / terminal_failure_penalty**：`info` 为空，无显式成功/失败标志，不可使用。
- **action/energy cost**：v1 阶段暂不引入燃料节省惩罚，避免 agent 不敢使用姿态引擎或主引擎，导致任务无法完成。后续迭代再逐步加入。
- **强姿态门控或动态课程权重**：v1 使用静态组件即可覆盖基本需求，复杂门控和课程参数留到后续版本。
- **original_reward **：明确禁止使用。

## 为什么不使用终端奖励
环境不提供 `info` 字段，无法可靠获取 episode 终止原因，因此不依赖任何终端成功/失败信号。所有奖励都从当前步和下一步的观测中计算。

## 留到后续迭代的职责
- **燃料效率/动作惩罚**：惩罚非零动作，鼓励最少推力完成着陆。
- **速度/姿态动态约束**：随着 training_progress 引入更严格的速度上限或姿态 gate。
- **更精细的软着陆条件**：利用环境中可能存在的其他隐性信号（例如归一化地面接触力）来进一步降低硬着陆风险。

## 训练后应观察的 failure modes
- **violent touchdown**：agent 可能学会快速冲向垫子，忽略姿态和速度，导致 crash 终止并突然获得很大的负 progress，但训练早期可能暂时幸运；需要监控 landing_bonus 平均大小和 episode 终止原因。
- **hover / loiter**：过度惩罚速度或角度可能使 agent 不敢移动，在远离目标处保持悬浮。应观察 distance_progress 是否持续为正。
- **leg-contact reward hacking**：agent 可能学会用单腿反复轻碰垫子刷奖励，但 episode 不会终止；需要检查终止时的 real success 比例（通过外部评估）。
- **姿态惩罚比例失调**：若角度惩罚太大，agent 可能完全避免任何旋转，导致无法修正方向。观察 angle 分布是否在合理范围内。