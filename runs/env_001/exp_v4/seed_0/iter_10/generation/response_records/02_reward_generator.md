# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward - 奖励每一步更接近目标
    # 目标位置在 (0, 0)，因为 obs[0] 和 obs[1] 是相对于目标平台的坐标
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta  # 权重10，鼓励向目标移动

    # 稳定/安全约束：stability_penalty - 惩罚高速、大姿态角和大角速度
    # 使用 next_obs 的状态，因为奖励基于动作后的结果
    vel_penalty = -0.5 * (next_obs[2] ** 2 + next_obs[3] ** 2)  # 惩罚线速度
    angle_penalty = -0.3 * (next_obs[4] ** 2)  # 惩罚姿态角偏差
    angvel_penalty = -0.2 * (next_obs[5] ** 2)  # 惩罚角速度
    stability_penalty = vel_penalty + angle_penalty + angvel_penalty

    # 任务完成 proxy：soft_landing_proxy - 当接近目标、低速、小姿态角且双接触时给予小奖励
    near_target = next_dist < 0.5
    low_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5 < 0.3
    stable_angle = abs(next_obs[4]) < 0.2
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 5.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 效率/动作代价：energy_penalty - 小权重惩罚使用引擎
    # action 0: no_engine, 1: left_orientation, 2: main_engine, 3: right_orientation
    engine_used = 1.0 if action != 0 else 0.0
    energy_penalty = -0.1 * engine_used

    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty

    # 构建组件字典
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward**（主学习信号）：基于距离变化量的密集过程奖励，鼓励智能体每一步都更接近目标位置。使用 `10.0 * (current_dist - next_dist)` 的形式，直接奖励向目标移动的行为。

2. **stability_penalty**（稳定约束）：惩罚高速（线速度平方）、大姿态角（角度平方）和大角速度（角速度平方），权重分别为 -0.5、-0.3、-0.2。这有助于智能体学会平稳接近目标，避免高速撞击或姿态失控。

3. **landing_bonus**（任务完成 proxy）：当智能体同时满足接近目标（距离<0.5）、低速（速度<0.3）、小姿态角（<0.2弧度）和双接触（左右支撑均接触）时，给予 5.0 的额外奖励。这是一个软性的着陆成功代理信号。

4. **energy_penalty**（动作代价）：对任何非零动作（即使用引擎）施加 -0.1 的小惩罚，鼓励智能体减少不必要的引擎使用，优化燃料消耗。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

根据环境卡片，`explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，info 字典为空，无法区分成功（稳定着陆）和失败（坠毁/出界）的终止。因此，使用终端奖励会引入不确定性，可能导致错误的学习信号。我们选择通过密集的进度奖励和软性着陆代理来引导学习。

## 留到后续迭代的组件

- **time_penalty**：当前未加入，因为过早加入可能导致智能体为了节省时间而采取冒险行为。如果训练后观察到智能体在目标附近徘徊不前，可以小权重加入。
- **gated_reward**：当前未使用，因为门控机制可能过于复杂，且可能阻碍探索。后续若发现安全约束被进度奖励抵消，可考虑加入。
- **更复杂的势能塑形**：当前使用简单的 progress_delta，后续可考虑 potential_based_shaping 以提供更标准的引导。

## 训练后应观察的 failure mode

1. **goal_near_oscillation**：智能体在目标附近来回震荡，无法稳定着陆。此时应检查 progress_reward 是否过大导致震荡，或 stability_penalty 是否不足。
2. **high_reward_without_success**：智能体获得高奖励但未成功着陆（如只在目标附近徘徊）。此时应收紧 landing_bonus 条件或增加稳定性惩罚。
3. **fast_crash_near_goal**：智能体快速冲向目标但坠毁。此时应增加速度惩罚或调整 progress_reward 权重。
4. **agent_afraid_to_move**：智能体因惩罚过大而不敢移动。此时应降低 stability_penalty 或 energy_penalty 的权重。
