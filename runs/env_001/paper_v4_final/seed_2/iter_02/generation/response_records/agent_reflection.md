# Response Record

# 设计理由
**修改组件**：`landing_bonus`（着陆奖励）——将其从硬性双脚接地门控改为连续软接触门控。

**问题诊断**：  
- 训练后记分显示 `landing_bonus` 的 `active_rate` 只有 **1.0%**，说明 agent 极少同时满足双脚接地条件，因此几乎接收不到与着陆相关的正向引导。  
- 全时二次惩罚（progress、pose_penalty）占据主导，agent 只能通过“不要偏离太多”来避免惩罚，形成一种“悬停/徘徊而后因超时或越界终止”的低分策略。  
- 连续化接触门控可大幅提高奖励密度，让 agent 在**单脚触地、接近低姿态误差**时就能获得部分奖励，从而更早地被引导至降落区域，并最终学会双脚着陆。

**数学形式**：  
- 原硬门：`both_leg_grounded = 1.0 if (left_contact>0.5 and right_contact>0.5) else 0.0`  
- 改为软门：`contact_factor = (left_contact + right_contact) / 2.0`（单脚 → 0.5，双脚 → 1.0）  
- 系数保持 `10.0`；整体公式不变：`10.0 * contact_factor * speed_factor * angle_factor`。  
- 当 agent 接近成功（低速、小角、双脚触地）时，仍可获得接近原公式的量级；单脚触地时可获得减半的奖励，足以产生学习梯度。

**系数校准**：  
- 当前主信号（progress）per-step 约 -0.036，pose_penalty 约 -0.018。  
- 若 agent 在平台附近获得 `contact_factor=0.5, speed_factor=0.6, angle_factor=0.6`，则 landing_bonus 约 `10*0.5*0.6*0.6 = 1.8`，远大于惩罚幅值，能有效驱动行为。  
- 这些因子组合（低速、小角、有接触）对应真正接近成功的状态，因此即使分量较大，也符合“终端奖励主导”的自然过渡。

**与其他组件的关联**：  
- 不需要额外修改惩罚或 progress 组件，本次仅激活着陆奖励的梯度，后续若出现悬停等问题再进一步调整。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs dimensions per environment card
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Primary progress signal: dense quadratic penalty on position and velocity errors.
    #    Encourages the lander to move toward (0,0) with zero speed.
    pos_sq_error = x_pos**2 + y_pos**2
    vel_sq_error = x_vel**2 + y_vel**2
    progress = -0.05 * pos_sq_error - 0.1 * vel_sq_error

    # 2. Stability constraint: quadratic penalty on body angle and angular velocity.
    #    Keeps the lander upright and prevents excessive spinning.
    pose_penalty = -5.0 * (body_angle**2) - 0.5 * (angular_vel**2)

    # 3. Soft landing bonus: now uses a continuous contact factor instead of a hard gate.
    #    contact_factor = (left_contact + right_contact) / 2.0, so single-leg contact gives 0.5.
    contact_factor = (left_contact + right_contact) / 2.0
    speed_magnitude = abs(x_vel) + abs(y_vel)
    speed_factor = 1.0 / (1.0 + 5.0 * speed_magnitude)
    angle_factor = 1.0 / (1.0 + 20.0 * abs(body_angle))
    landing_bonus = 10.0 * contact_factor * speed_factor * angle_factor

    total_reward = progress + pose_penalty + landing_bonus

    components = {
        'progress': progress,
        'pose_penalty': pose_penalty,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测维度已使用，信号无缺失；但 `landing_bonus` active_rate 仅 1%，属于僵尸组件。
- **behavior**: agent 长期悬停/徘徊，最终因超时或边界终止，未能向目标平台靠近。
- **signal**: 缺少连续的接近着陆奖励，导致稀疏反馈无法引导学习。
- **level**: Level 2（结构变换：二值门控 → 连续软因子）
- **hypothesis**: 连续化接触因子将使 agent 在单脚触地时即获得奖励，大幅提升 reward 密度，从而引导它学习降落到平台并最终完成双脚着陆。
- **risk**: agent 可能学会只让单脚触地来获取奖励而不完成最终双脚着陆；但单脚触地时仍需满足低速度、小角度，这本身已接近成功，后续可通过微调或引入双脚特定 bonus 解决。
