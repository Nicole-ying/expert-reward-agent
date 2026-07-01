# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 计算当前位置到目标(0,0)的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist  # 正值表示更接近目标
    progress_delta_reward = 10.0 * progress_delta  # 主信号权重较大

    # ========== 稳定约束：stability_penalty ==========
    # 惩罚速度、姿态角和角速度，鼓励稳定接近
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = abs(next_obs[4])  # 姿态角偏离0的绝对值
    angular_vel_penalty = abs(next_obs[5])  # 角速度绝对值
    stability_penalty = -0.5 * speed - 0.3 * angle_penalty - 0.1 * angular_vel_penalty

    # ========== 任务完成 proxy：soft_landing_proxy ==========
    # 当接近目标、低速、姿态稳定且双支撑接触时给予小奖励
    near_target = next_dist < 0.3
    low_speed = speed < 0.2
    stable_angle = abs(next_obs[4]) < 0.2
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    soft_landing_proxy = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # ========== 动作代价：energy_penalty（小权重） ==========
    # 使用引擎（action != 0）时给予小惩罚
    energy_penalty = -0.1 if action != 0 else 0.0

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy + energy_penalty

    # ========== 组件字典 ==========
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

1. **progress_delta_reward**（主学习信号，权重10.0）
   - 角色：引导智能体每一步都更接近目标位置(0,0)
   - 数学形式：`d(obs, goal) - d(next_obs, goal)`，正值奖励接近，负值惩罚远离
   - 选择理由：这是导航到达类任务最直接的过程引导信号，能提供密集且平滑的学习梯度

2. **stability_penalty**（稳定约束，权重-0.5/-0.3/-0.1）
   - 角色：惩罚高速、大姿态角和大角速度，鼓励稳定接近和着陆
   - 数学形式：`-0.5*speed - 0.3*|angle| - 0.1*|angular_velocity|`
   - 选择理由：任务要求"稳定在目标平台上"，且观察空间包含速度、姿态角、角速度，加入轻量稳定约束可防止高速撞击或姿态失稳

3. **soft_landing_proxy**（任务完成近似信号，条件奖励2.0）
   - 角色：当智能体同时满足"接近目标、低速、姿态稳定、双支撑接触"时给予小奖励，作为成功着陆的软代理
   - 选择理由：虽然没有显式success flag，但通过组合多个条件可以近似检测着陆成功，同时避免直接使用contact作为success（contact必须与位置、速度、姿态组合使用）

4. **energy_penalty**（动作代价，权重-0.1）
   - 角色：轻微惩罚使用引擎的动作，鼓励节能
   - 选择理由：任务要求"尽可能少地使用引擎推力"，但权重很小（-0.1）以避免智能体不敢移动

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available=false`：info字典为空，没有显式success flag，无法可靠判断成功终止
- `explicit_failure_flag_available=false`：同样没有显式failure flag，无法区分失败终止
- 使用这些终端奖励会诱导LLM发明不存在的info字段，违反设计原则

## 留到后续迭代的组件

- **time_penalty**：当前任务没有时间限制，且加入时间惩罚可能导致智能体冒险或快速失败，留到能稳定接近后再考虑
- **gated_reward**：复杂门控机制，v1阶段不需要
- **dynamic curriculum**：训练进度相关调整，后续迭代再引入
- **terminal_success_reward / terminal_failure_penalty**：当wrapper明确暴露success/failure信号后再加入

## 训练后应观察的failure mode

1. **goal_near_oscillation**：智能体在目标附近震荡但不完成着陆——观察progress_delta_reward是否在目标附近变小，soft_landing_proxy是否难以触发
2. **high_reward_without_success**：智能体获得高奖励但未成功着陆——检查是否过度依赖progress_delta_reward而忽略稳定约束
3. **fast_crash_near_goal**：接近目标时高速撞击——观察stability_penalty是否足够抑制高速
4. **agent_afraid_to_move**：智能体不敢移动——检查energy_penalty是否过大，或stability_penalty是否过强
5. **contact reward hacking**：智能体通过非正常方式触发contact——观察soft_landing_proxy的条件是否过于宽松
