# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Compute Euclidean distance to target (landing pad center)
    d_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Main learning signal: progress towards the target
    progress = d_curr - d_next   # positive when moving closer
    # Clip progress to avoid huge single-step rewards (safe range for 2D normalised-like coordinates)
    progress_clipped = max(-0.5, min(0.5, progress))
    progress_reward = 10.0 * progress_clipped

    # Stability penalty: discourage high speed, large angle and angular velocity
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    # Vertical velocity is penalised less to allow gentle descent
    stability_penalty = -0.5 * abs(vx) - 0.2 * abs(vy) - 0.5 * abs(angle) - 0.1 * abs(angular_vel)

    # Soft landing proxy: dense reward when very close, slow, upright and both legs in contact
    dist_thresh = 0.05
    vel_thresh = 0.1
    angle_thresh = 0.05
    angvel_thresh = 0.05
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    if (d_next < dist_thresh and
        abs(vx) < vel_thresh and
        abs(vy) < vel_thresh and
        abs(angle) < angle_thresh and
        abs(angular_vel) < angvel_thresh and
        left_contact > 0.5 and
        right_contact > 0.5):
        landing_bonus = 1.0
    else:
        landing_bonus = 0.0

    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件
1. **progress_reward**（主学习信号）  
   每一步根据欧氏距离的减少量给予奖励，鼓励飞行器快速接近着陆垫。使用裁剪防止单步异常值，并提供连续的梯度信号。

2. **stability_penalty**（稳定约束）  
   惩罚过大的水平速度、垂直速度、机体倾斜角和角速度。垂直速度惩罚系数较低，允许温和下降，避免agent“悬停不敢落”。

3. **landing_bonus**（任务完成近似信号）  
   当飞行器同时满足非常接近目标、低速度、姿态水平、角速度极小且双支撑腿接触时，每步给予固定奖励。这是一个稠密的proxy，弥补环境没有显式成功标志的不足，引导agent完成最后着陆。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境`info`为空字典，没有显式成功或失败标记，因此不能依赖`info['success']`等字段。所有终止条件（如`body_not_awake_or_settled`）都是上下文相关的，单独使用会导致误判。

## 后续迭代计划
- **energy_penalty**（主引擎使用惩罚）：待agent能稳定着陆后，以极小权重加入，实现“尽量少用主引擎推力”的节能要求。
- **更精细的稳定约束**（如仅惩罚向上速度、下降速度阈值）：若观察到不敢下降或快速坠毁，可迭代调整。
- **gated_reward / dynamic curriculum**：如需更安全的着陆阶段，可在后续引入。

## 训练后应观察的 failure mode
- **目标附近振荡**：progress_reward可能驱使agent来回摆动而不完成着陆；观察距离/速度震荡是否过大。
- **悬停不落**：stability_penalty对垂直速度的惩罚可能让agent不敢下降，应检查是否长期维持在低空不接触。
- **landing_bonus 伪造**：agent可能学习只触碰一条腿或接触但姿态异常来骗取奖励，需要检查条件是否过于宽松。
- **总奖励被惩罚项支配**：若稳定性惩罚过大，agent可能变得保守，应监控各组件均值，必要时调整权重。
