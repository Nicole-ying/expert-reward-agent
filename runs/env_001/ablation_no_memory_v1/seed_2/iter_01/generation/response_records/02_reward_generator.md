# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander reaching a central target pad.
    
    obs: [x, y, vx, vy, angle, angular_vel, left_contact, right_contact]
    next_obs: same structure after taking action
    """
    # --- Unpack observations ---
    x, y = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    omega_next = next_obs[5]
    left_contact_next, right_contact_next = next_obs[6], next_obs[7]

    # --- 1. Main learning signal: progress towards the target pad ---
    dist_current = (x**2 + y**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress_reward = dist_current - dist_next   # positive if getting closer

    # --- 2. Light stability penalty: suppress extreme velocity / angle / angular velocity ---
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    stability_penalty = -0.01 * speed_next - 0.02 * abs(angle_next) - 0.02 * abs(omega_next)

    # --- 3. Soft landing proxy: bonus when close to pad with good conditions ---
    # Conditions: near target, low speed, upright, both legs in contact
    near_target = abs(x_next) < 0.15 and abs(y_next) < 0.3
    low_speed = speed_next < 0.5
    upright = abs(angle_next) < 0.3
    both_contact = (left_contact_next == 1.0) and (right_contact_next == 1.0)

    landing_bonus = 0.5 if (near_target and low_speed and upright and both_contact) else 0.0

    # --- Total reward ---
    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward（主学习信号）**  
   - 角色：密集驱动力，引导飞行器每一步向目标着陆垫中心靠近。  
   - 计算方式：当前步到目标的距离减去下一步到目标的距离，正值表示接近。利用 obs[0]、obs[1] 和 next_obs[0]、next_obs[1] 提供每步梯度。  
   - 为什么选它：这是 navigation_goal_reaching 任务最直接的连续信号，不依赖 sparse 事件，与目标位置强相关。

2. **stability_penalty（轻量稳定/安全约束）**  
   - 角色：抑制过大的线速度、机体倾角和角速度，帮助飞行器学习平稳飞行和最终稳定着陆。  
   - 权重设计：数值相比 progress_reward 小（系数 0.01 / 0.02），不会压制探索，但能提供次级信号使行为更稳定。  
   - 为什么加入：该任务要求姿态平稳、安全接触，过大的速度或倾角容易导致 crash 或超出视口，约束可降低这些风险，同时为后续迭代铺设基础。

3. **landing_bonus（任务完成近似信号 proxy）**  
   - 角色：多条件组合的“完成着陆”奖励，引导飞行器在接近目标后执行最终的稳定停靠动作。  
   - 条件：在目标附近（|x|<0.15, |y|<0.3）、低线速（<0.5）、姿态竖直（|angle|<0.3）、双腿同时接触着陆垫。全部满足才给予 0.5 奖励。  
   - 为什么加入：缺少显式 success flag，且 progress_reward 在目标附近梯度消失，可能造成震荡而不着陆。这个 proxy 给出明确的“停稳并接触”信号，帮助完成最终行为。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡片明确指出 `explicit_success_flag_available: false` 且 `explicit_failure_flag_available: false`，info 字典为空。  
- 不存在任何客观的 success/failure 标志，因此不能在 reward 中使用 terminal 的稀疏成功/失败奖励。  
- 终止条件（crash/出界/睡眠）混在一起且未提供原因，不可用于可靠的分界奖励。

## 未在 v1 使用的组件及后续迭代规划

- **energy_penalty / time_penalty**：当前版本不引入。agent 应首先学会到达并稳定着陆，后续迭代（v2/v3）再添加主引擎使用惩罚或时间惩罚以优化效率。  
- **gated_reward / dynamic_curriculum**：复杂门控留到后期，当需要解决特定 failure mode 时再考虑。  
- **更复杂的 stability 项**（如方向加速度约束）：如果 v1 训练后频繁出现高速失稳碰撞，可适当增大 stability_penalty 权重或加入更多维度。

## 训练后应观察的 failure mode

1. **目标附近震荡（high_reward_without_success）**  
   agent 可能学会在目标垫上方往复运动而不着陆，因为 progress_reward 仍然可以提供微小的正奖励。应检查训练后期 agent 是否频繁出现在目标附近但不触发 landing_bonus 的情况。  
2. **过早睡眠（spurious settle）**  
   终止条件之一 `body_not_awake_or_settled` 可能在不正确的位置触发，导致 episode 提前结束。agent 可能学到在远处就减速休眠来避免 crash 惩罚，需要观察是否出现 reward 不低但从未到达目标的现象。  
3. **快速 crash 近目标**  
   如果 progress_reward 权重相对稳定性惩罚过大，agent 可能高速冲向目标但因控制不稳而 crash。应监控碰撞率和到达目标的稳定性。  
4. **landing_bonus 触发率不足**  
   若条件过严，proxy 可能几乎不触发，导致最终行为仍不明确；可在后续迭代中逐步放宽阈值（如扩大 near_target 范围）并确认不会诱导 exploit。  
5. **动作过于保守（agent afraid to move）**  
   如果 stability_penalty 系数设得过大，agent 可能倾向不做大动作，导致探索不足，无法到达目标。需对比训练初期的 episode 长度和距离减少趋势。  

所有组件设计均遵循可用信号原则，未使用任何未声明的 info 字段或 obs 切片，且不依赖官方 original_reward。下一步将根据训练曲线和这些 failure mode 分析迭代优化。
