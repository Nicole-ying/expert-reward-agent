# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 1. 主学习信号: progress_delta_reward
    #    基于当前位置到目标(0,0)的距离变化，引导飞行器接近目标
    # ============================================================
    # 当前距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    # 下一步距离
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    # 距离减少为正奖励，增加为负奖励
    progress_delta = current_dist - next_dist
    # 缩放因子，使奖励值在合理范围
    progress_scale = 2.0
    progress_delta_reward = progress_delta * progress_scale

    # ============================================================
    # 2. 稳定/安全约束: stability_penalty
    #    惩罚高速、大姿态角和大角速度，鼓励稳定接近
    # ============================================================
    # 速度惩罚（使用next_obs，因为动作执行后的状态）
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_penalty_weight = 0.1
    speed_penalty = -speed_penalty_weight * speed

    # 姿态角惩罚（角度偏离0度）
    angle_penalty_weight = 0.05
    angle_penalty = -angle_penalty_weight * abs(next_obs[4])

    # 角速度惩罚
    angular_vel_penalty_weight = 0.02
    angular_vel_penalty = -angular_vel_penalty_weight * abs(next_obs[5])

    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty

    # ============================================================
    # 3. 任务完成proxy: soft_landing_proxy
    #    当飞行器接近目标、速度低、姿态稳定且双支撑接触时给予小奖励
    # ============================================================
    # 条件：距离目标很近（<0.5）
    near_target = 1.0 if next_dist < 0.5 else 0.0
    # 条件：速度很低（<0.3）
    low_speed = 1.0 if speed < 0.3 else 0.0
    # 条件：姿态角很小（<0.2弧度）
    stable_angle = 1.0 if abs(next_obs[4]) < 0.2 else 0.0
    # 条件：双支撑接触
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    # 所有条件满足时给予小奖励
    landing_bonus_weight = 1.0
    soft_landing_proxy = landing_bonus_weight * near_target * low_speed * stable_angle * both_contact

    # ============================================================
    # 4. 动作代价: energy_penalty（小权重）
    #    轻微惩罚使用引擎，鼓励节能
    # ============================================================
    # action 0: no_engine -> 无惩罚
    # action 1,2,3: 使用引擎 -> 小惩罚
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * engine_use

    # ============================================================
    # 总奖励
    # ============================================================
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy + energy_penalty

    # 组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号）
   - 角色：引导飞行器向目标（0,0）移动，每一步奖励距离的减少
   - 权重：缩放因子2.0，使奖励值在合理范围（典型距离变化0~2，奖励0~4）
   - 优势：密集、平滑、直接反映任务进展

2. **stability_penalty**（稳定约束）
   - 角色：惩罚高速、大姿态角和大角速度，鼓励稳定接近
   - 权重：速度0.1、姿态角0.05、角速度0.02，总惩罚通常不超过-0.5
   - 优势：防止飞行器高速冲过目标或姿态失控

3. **soft_landing_proxy**（任务完成近似信号）
   - 角色：当飞行器接近目标、低速、稳定且双支撑接触时给予小奖励
   - 权重：1.0，但条件严格，只有真正着陆时才会激活
   - 优势：提供稀疏但明确的着陆信号，帮助学习最终稳定

4. **energy_penalty**（动作代价）
   - 角色：轻微惩罚使用引擎，鼓励节能
   - 权重：0.01，非常小，不会阻碍学习
   - 优势：引导智能体减少不必要的引擎使用

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡片明确说明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- info字典为空，没有任何成功/失败标志可用
- 使用这些项会诱导LLM发明不存在的info字段，违反设计原则

## 留到后续迭代的组件

- **time_penalty**：当前任务没有时间限制，过早加入可能导致冒险行为
- **gated_reward**：复杂门控逻辑，v1应保持简单
- **terminal_success_reward / terminal_failure_penalty**：需要wrapper暴露成功/失败标志后再加入
- **dynamic curriculum**：v1不需要课程学习

## 训练后应观察的failure mode

1. **goal_near_oscillation**：飞行器在目标附近震荡但不稳定着陆
   - 观察：progress_delta_reward接近0但soft_landing_proxy不激活
   - 对策：增加stability_penalty权重或调整soft_landing_proxy条件

2. **high_reward_without_success**：获得高奖励但未真正着陆
   - 观察：progress_delta_reward高但soft_landing_proxy为0
   - 对策：收紧soft_landing_proxy条件或增加着陆bonus

3. **fast_crash_near_goal**：高速冲向目标然后坠毁
   - 观察：stability_penalty负值大但progress_delta_reward正
   - 对策：增加速度惩罚权重或加入速度上限约束

4. **agent_afraid_to_move**：智能体不敢使用引擎
   - 观察：energy_penalty导致action始终为0
   - 对策：降低energy_penalty权重或移除该组件