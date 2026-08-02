# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations (all are scalar float)
    x, y, vx, vy, angle, angvel, l_contact, r_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nl_contact, nr_contact = next_obs

    # 1. Main progress signal: distance reduction to target pad (improvement_delta)
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_obs - dist_next  # positive when getting closer

    # 2. Stability constraints (quadratic penalties on next state)
    angle_penalty = -0.1 * (nangle ** 2)        # penalize tilt
    angvel_penalty = -0.05 * (nangvel ** 2)     # penalize rotation

    # 3. Soft landing proxy (joint_condition_proxy using bounded factors)
    v_thresh = 0.2
    contact = nl_contact * nr_contact                 # 0 or 1
    factor_vx = max(0.0, 1.0 - abs(nvx) / v_thresh)  # [0,1], 1 when speed near 0
    factor_vy = max(0.0, 1.0 - abs(nvy) / v_thresh)  # [0,1]
    landing_bonus = contact * factor_vx * factor_vy   # in [0,1], only when settled

    # Weights
    w_progress = 10.0
    w_landing  = 20.0

    total = (w_progress * progress +
             angle_penalty + angvel_penalty +
             w_landing * landing_bonus)

    components = {
        "progress": w_progress * progress,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_bonus": w_landing * landing_bonus
    }

    return float(total), components
```

# reward_v1 设计说明

## 任务画像与动力学
- **task_family**：`navigation_goal_reaching`
- **dynamics_subtype**：`goal_approach_and_soft_contact`（2D 着陆器，离散控制）
- **核心目标**：尽快安全降落在中央目标垫上，次要目标为节能

## 选定的奖励角色
| Role | 信号来源 | 公式算子 | 理由 |
|---|---|---|---|
| **主进展信号** | `x_position`, `y_position` (相对目标) | `improvement_delta` (`dist_obs - dist_next`) | 稠密、每步梯度，直接反映朝目标靠近的程度；避免依赖稀疏终止事件 |
| **姿态/角速度约束** | `body_angle`, `angular_velocity` (next_obs) | `quadratic_penalty` | 防止翻滚和过大倾角导致失控，但不使用 hinge 以避免过度惩罚安全范围内的微小偏移 |
| **软着陆近似** | `left_support_contact`, `right_support_contact`, `x_velocity`, `y_velocity` (next_obs) | `joint_condition_proxy`（乘积 + bounded 因子） | 当双脚接触目标垫且速度趋近零时提供高额奖励，引导 agent 完成最终“settle”动作；连续因子保证平滑梯度 |

## 排除的角色与原因
- **terminal_success_reward / terminal_failure_penalty**：info 为空，不可靠的推断信号曾导致历史尝试分数暴跌（-61 至 -122）；v1 完全放弃硬终端信号
- **action_efficiency**：离散动作空间且历史加入动作惩罚后性能显著恶化（-61，-122），表明当前阶段不宜压制探索
- **soft_health_gate / preview_conditioned_reward**：无前瞻传感器，健康门控可后续考虑
- **explicit stability hinge**：为避免过早引入硬阈值，留待后续迭代优化

## 为何不使用终端成功/失败奖励
环境没有显式成功/失败标志，从观测推断的条件复杂且容易产生错误信号；历史尝试证明这类奖励导致严重的负面性能。本设计完全依赖稠密的进展信号和连续约束来塑造行为，从而确保每一步都能获得有意义的反馈。

## 留给后续迭代的职责
- **动态课程**：随着训练进度逐渐收紧速度/姿态阈值
- **效率/能量代价**：如果后期观察到不必要的发动机频繁点火再引入
- **更强的着陆条件**：当 baseline 成功着陆率 > 50% 后，可加入垂直速度上限的 hinge 惩罚

## 预期的训练后失败模式
- **高速冲撞**：agent 可能学会快速接近目标但减速不足，导致 crash（即使有角度惩罚，水平/垂直速度过大仍可能导致撞地或出界）；后续可增加基于高度的速度上限门控
- **原地弹跳**：landing_bonus 可能诱导 agent 反复点触垫面以刷分，此时需检查接触奖励是否被滥用，可增加“持续接触”要求
- **姿态振荡**：若角度和角速度惩罚过轻，agent 可能通过来回摆荡保持方向，但可能失去控制；如果出现此现象，需稍增 penalty 权重或引入角加速度惩罚
- **过早着陆尝试**：agent 可能在半空就尝试双脚着垫信号（实际不会发生，因为接触只在脚触地时置 1），风险很小；但若有类似 exploit，可加入距离因子限制
