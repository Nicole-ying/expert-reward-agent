# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. Main learning signal: progress_delta_reward
    # Reward getting closer to the target (origin in relative coordinates)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. Stability penalty: penalize high velocity, large angle, and high angular velocity
    # Use next_obs to penalize the state after taking action
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_body_angle)
    speed_penalty = 0.3 * speed
    ang_vel_penalty = 0.2 * abs(next_ang_vel)
    stability_penalty = -(angle_penalty + speed_penalty + ang_vel_penalty)
    
    # 3. Soft landing proxy: small bonus when near target, low speed, stable angle, and both contacts
    near_target = next_dist < 0.5
    low_speed = speed < 0.3
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    soft_landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0
    
    # 4. Small distance anchor to prevent drifting far away
    distance_anchor = -0.1 * next_dist
    
    # Combine components
    total_reward = progress_reward + stability_penalty + soft_landing_bonus + distance_anchor
    
    # Build components dict
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号，权重10.0）
   - 角色：引导智能体每一步都更接近目标位置（原点）。
   - 数学形式：`current_dist - next_dist`，正值表示靠近目标。
   - 这是核心学习信号，提供密集的、与目标距离变化直接相关的奖励。

2. **stability_penalty**（稳定约束，权重组合）
   - 角色：惩罚高速、大姿态角和高角速度，鼓励稳定飞行。
   - 包含三个子项：角度惩罚（0.5×|angle|）、速度惩罚（0.3×speed）、角速度惩罚（0.2×|ang_vel|）。
   - 使用 `next_obs` 中的状态，确保惩罚的是动作执行后的结果。

3. **soft_landing_bonus**（任务完成代理，权重2.0）
   - 角色：当智能体同时满足接近目标、低速、稳定姿态和双支撑接触时，给予小奖励。
   - 这是一个软代理信号，不是真正的 success flag，但能鼓励正确的着陆行为。
   - 条件组合（near_target + low_speed + stable_angle + both_contact）防止 contact hacking。

4. **distance_anchor**（辅助锚点，权重-0.1）
   - 角色：小权重惩罚远离目标，防止智能体完全忽略距离。
   - 权重很小，不会干扰主信号，但提供额外的稳定性。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，没有可靠的 success/failure 信号。
- 终止条件（`body_not_awake_or_settled`、`crash_or_body_contact`）含义不明确，无法区分成功着陆和失败坠毁。
- 使用 terminal 奖励可能导致误判，误导智能体学习错误行为。

## 留到后续迭代的组件

- **energy_penalty**：当前权重为0，因为过早加入可能导致智能体不敢使用引擎，无法学习基本接近行为。待智能体能稳定接近目标后再加入。
- **time_penalty**：当前未使用，因为可能鼓励冒险行为。待智能体能成功着陆后再考虑加入以优化效率。
- **gated_reward**：当前未使用，因为门控条件难以定义且可能阻碍学习。
- **terminal_success_reward / terminal_failure_penalty**：待 wrapper 明确暴露 success/failure 信号后再加入。

## 训练后应观察的 failure mode

1. **目标附近震荡**：如果 progress_delta_reward 导致智能体在目标附近来回移动而不着陆，需要调整稳定性惩罚或加入时间惩罚。
2. **高速撞击**：如果智能体以高速接近目标但无法减速，需要增大速度惩罚权重。
3. **姿态失稳**：如果智能体在接近目标时姿态角过大，需要增大角度惩罚权重。
4. **contact hacking**：如果智能体学会通过非正常方式触发接触标志，需要收紧 soft_landing_bonus 的条件或降低其权重。
5. **高奖励但未成功**：如果 progress_reward 很高但智能体从未触发 soft_landing_bonus，说明需要调整着陆代理条件或增加其他引导信号。