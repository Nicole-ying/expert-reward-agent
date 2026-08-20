# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步观测
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # 下一步观测
    next_x, next_y, next_vx, next_vy, next_angle, next_ang_vel, next_left_contact, next_right_contact = next_obs

    # 1. 主进展：鼓励向目标垫 (0,0) 靠近（距离减小）
    old_dist = (x ** 2 + y ** 2) ** 0.5
    new_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = old_dist - new_dist          # >0 接近，<0 远离
    w_dist = 1.0
    progress_reward = w_dist * progress

    # 2. 靠近目标时的速度惩罚：强制减速，防止高速撞击
    closeness = max(0.0, 1.0 - new_dist / 1.5)    # 距离<1.5时激活，距离0时=1
    speed = abs(next_vx) + abs(next_vy)
    w_speed = 0.5
    speed_penalty = -w_speed * closeness * speed

    # 3. 角速度轻量惩罚：保持姿态稳定
    w_angvel = 0.1
    angular_vel_penalty = -w_angvel * abs(next_ang_vel)

    # 4. 着陆奖励：双脚接触 + 位置对中 + 低速 + 小角度
    x_thresh = 0.3
    y_thresh = 0.3
    vel_thresh = 0.5
    angle_thresh = 0.2  # rad ≈ 11°

    contact_factor = next_left_contact * next_right_contact  # 0 或 1
    pos_x_factor = max(0.0, 1.0 - abs(next_x) / x_thresh)
    pos_y_factor = max(0.0, 1.0 - abs(next_y) / y_thresh)
    vel_factor = max(0.0, 1.0 - (abs(next_vx) + abs(next_vy)) / vel_thresh)
    angle_factor = max(0.0, 1.0 - abs(next_angle) / angle_thresh)

    landing_multiplier = (contact_factor * pos_x_factor * pos_y_factor
                          * vel_factor * angle_factor)
    w_land = 2.0
    landing_reward = w_land * landing_multiplier

    total_reward = (progress_reward + speed_penalty
                    + angular_vel_penalty + landing_reward)

    components = {
        "progress": progress_reward,
        "speed_penalty": speed_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "landing": landing_reward
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- `navigation_goal_reaching`，子类型 `goal_approach_and_soft_contact`，动作空间离散。

## selected reward roles
根据 environment_card 的 `reward_role_decomposition`，v1 选择了：

1. **approach_target**（主学习信号）  
2. **stable_landing**（任务完成近似信号）  
另外加入两个轻量稳定约束：近端速度惩罚和全局角速度惩罚（健康/安全角色），对应 expert_reward_context 中的“健康/稳定约束”类。

## role_to_signal_mapping
- `approach_target` → `x_position, y_position` 的欧氏距离，使用 **improvement_delta**：`old_dist - new_dist`。
- 近端速度约束 → `x_velocity, y_velocity`，用接近度作为 **soft_health_gate** 乘线性速度惩罚。
- 姿态稳定 → `angular_velocity`，直接使用 **dense_state_signal (penalty)**：`-w * |ang_vel|`。
- `stable_landing` → 双脚接触 `left_support_contact, right_support_contact`、位置 `x_position, y_position`、平动速度、`body_angle`，用 **joint_condition_proxy**（乘积形式）构造着陆软奖励。

## formula operator 选用理由
- **improvement_delta** 用于主进展：直接衡量距离缩短，比绝对位置奖励更能防止停滞；与速度奖励分离，避免 agent 将速度本身误认为进展。
- **soft_health_gate** 用于速度惩罚：仅在靠近目标时逐渐激活，早期允许正常机动，晚期强制减速，防止高速撞击。
- **dense_state_signal (penalty)** 用于角速度：常数轻度惩罚，抑制无谓的振荡旋转。
- **joint_condition_proxy** 用于着陆：乘积形式要求所有条件同时满足，避免单个接触信号带来的 reward hacking；各因子采用线性衰减 `1 - error/threshold`，保证梯度连续。

## excluded roles 及原因
- **action_penalty**：v1 阶段不加，优先学会到达和着陆；过早引入燃料成本可能抑制必要的发动机使用。
- **time_efficiency (常数负奖励)**：环境步数未暴露且易诱导 agent 提前 crash，不符合训练稳定性要求。
- **terminal_success_reward / terminal_failure_penalty**：info 字典为空，`explicit_success_flag_available=false`，无法构造可靠的终端事件；且从观测推断失败缺少明确边界，v1 改用连续约束替代硬终端信号。

## 后续迭代方向
- 如果 agent 出现巨大冲量但随后仍能稳定着陆，可考虑加入高度相关的势能 shaping 或更复杂的门控。
- 若着陆后 episode 不自动终止，且 agent 在着陆后仍维持奖励，可后续增加 episode 长度相关信息或轻量时间惩罚。
- 动作效率惩罚可在学习成熟后以极小权重（<0.01）引入，以减少不必要的发动机激活。

## 训练后应观察的 failure modes
- **高速撞击目标垫**：进度为正，但速度惩罚可能不够 → 检查 speed_penalty 的平均幅度。
- **长时间悬停不下降**：progress 接近 0，因为 y_position 不减小 → 可能需增强深度相关的进展分量或减小速度惩罚门限。
- **到目标附近后摆动不停**：landing 奖励持续较低，角速度/速度频繁波动 → 关注 contact_factor 和 vel_factor 的 episode 均值，考虑收紧角度或速度阈值。
- **单脚着陆倾斜**：landing_reward 在 contact_factor 上看可能为 0，导致 reward 被抑制 → 正常，应迫使 agent 同时接触。
