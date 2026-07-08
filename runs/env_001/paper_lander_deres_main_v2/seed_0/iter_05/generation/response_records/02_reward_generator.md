# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    奖励函数设计：鼓励飞行器平稳接近并双脚着陆目标平台。
    使用联合质量函数 + 接触放大因子，替代进度差奖励。
    """
    # 从 next_obs 提取状态变量
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 计算关键量
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    abs_angle = abs(body_angle)
    abs_angvel = abs(angular_vel)

    # 连续性落地质量函数：同时考量距离、速度、姿态
    alpha_dist = 0.5
    alpha_speed = 0.5
    alpha_angle = 2.0
    alpha_angvel = 1.0

    quality_exp_term = -(
        alpha_dist * distance +
        alpha_speed * speed +
        alpha_angle * abs_angle +
        alpha_angvel * abs_angvel
    )
    quality = 2.718281828 ** quality_exp_term   # exp(negative) ∈ (0, 1]

    # 接触信息：双脚同时接触为1，否则0
    both_contact = left_contact * right_contact   # 乘积：仅当两者都为1时为1

    # 接触放大奖励：在高质量状态下双脚接触获得额外加成
    k_contact = 0.5
    contact_bonus = k_contact * quality * both_contact

    total_reward = quality + contact_bonus

    components = {
        "landing_quality": quality,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 奖励组件与角色

1. **landing_quality（主学习信号）**  
   - 连续性指数函数，综合惩罚距离、速度、角度和角速度。  
   - 每一步都提供梯度，引导飞行器同时靠近目标、降低速度和保持姿态稳定。  
   - 不再依赖“上一步到下一步的进度差”，而是直接评价当前状态与理想着陆状态的相似度。

2. **contact_bonus（任务完成近似信号）**  
   - 在 landing_quality 基础上，双脚同时接触地面时，奖励值放大（×1.5）。  
   - 作为连续性的 soft proxy，避免离散二值条件被 exploit；只有着陆质量高时接触才会带来可观收益。  
   - 通过乘积形式与 quality 绑定，防止“轻触即得分”的取巧行为。

## 为何未使用 terminal_success_reward / terminal_failure_penalty

- 环境未提供显式的成功或失败标志（explicit_success_flag_available = false），info 字典为空。  
- 强行发明 info 中的 success / failure 字段会使奖励函数不可靠，违背设计约束。  
- 本设计通过 landing_quality 与 contact_bonus 组合间接传递完成信号的语义，且每一步都有稠密梯度的。

## 与之前失败方案的差异

- 之前尝试的 `progress_delta + soft_landing_proxy + stability_penalty` 组合停滞在低分区域。  
- 新方案将核心从“向目标靠近的变化量”转为“当前状态的联合质量评价”，并用乘积放大接触信号，避免二值条件带来的振荡或冻结。  
- 不包含显式的稳定性惩罚项，因为速度和角度已经融入质量函数中，减少了信号冲突和重复。

## 后续迭代预留

- **效率/燃料代价（energy_penalty）**：待 Agent 学会平稳着陆后再小权重加入，防止早期不敢点火。  
- **更复杂的成功门控（gated_reward）或动态课程**：如果出现“悬停但不完成着陆”的现象，可在后续版本加入更多条件约束。  
- **终端成功/失败奖励**：若环境未来提供显式终止原因或成功标志，可补充稀疏的 terminal reward 以加速收敛。

## 训练后应观察的 failure mode

- **悬停不落**：Agent 在高处保持小速度和小角度，距离不大但始终不接触平台，此时 landing_quality 较高而 contact_bonus 为零。观察是否需要调整 k_contact 或引入高度/接触的额外引导。  
- **缓慢下降但极度省能**：由于没有时间惩罚或速率下限，Agent 可能以极慢速度下降，导致 episode 变长。后续可根据需要加入弱时间惩罚。  
- **高速撞击后弹起**：尽管速度项会惩罚高速，但若权重不足仍可能出现。观察是否需要增大 alpha_speed 或加入速度阈值惩罚。
