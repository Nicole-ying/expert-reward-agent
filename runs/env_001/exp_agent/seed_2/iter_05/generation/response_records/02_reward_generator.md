# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 使用 obs[0], obs[1] 和 next_obs[0], next_obs[1] 计算到目标(0,0)的距离变化
    # 目标位置是 (0, 0)，因为 obs 已经是相对于目标平台的坐标
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist  # 正数表示更接近目标
    progress_delta_reward = 10.0 * progress_delta  # 权重10，鼓励每一步都更接近

    # ========== 稳定/安全约束：stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度，鼓励稳定接近和着陆
    # 使用 next_obs 的状态，因为这是动作执行后的结果
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = abs(next_obs[4])  # 姿态角偏离0的绝对值
    angular_vel_penalty = abs(next_obs[5])  # 角速度绝对值
    
    # 权重：速度惩罚0.5，姿态角惩罚0.3，角速度惩罚0.2
    stability_penalty = -0.5 * speed - 0.3 * angle_penalty - 0.2 * angular_vel_penalty

    # ========== 任务完成 proxy：soft_landing_proxy ==========
    # 当飞行器接近目标、速度低、姿态稳定且两个支撑接触时给予小奖励
    # 这是对成功着陆的软近似，不是真正的 success flag
    near_target = (next_dist < 0.5)  # 距离目标小于0.5
    low_speed = (speed < 0.3)  # 速度小于0.3
    stable_angle = (abs(next_obs[4]) < 0.2)  # 姿态角小于0.2弧度
    both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)  # 两个支撑都接触
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0  # 小权重奖励，避免过度激励

    # ========== 动作代价：energy_penalty（小权重） ==========
    # 惩罚使用引擎，鼓励节能
    # action 0: no_engine, action 1: left_orientation, action 2: main, action 3: right_orientation
    engine_use = 0.0
    if action == 1 or action == 3:
        engine_use = 0.5  # 姿态发动机
    elif action == 2:
        engine_use = 1.0  # 主发动机
    energy_penalty = -0.1 * engine_use  # 权重很小，避免agent不敢动

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号，权重10.0）
   - 角色：引导飞行器每一步都更接近目标位置(0,0)
   - 使用当前距离与下一步距离的差值作为奖励，正数表示进步
   - 这是最直接的导航学习信号，符合 navigation_goal_reaching 任务路由

2. **stability_penalty**（稳定约束，权重-0.5/-0.3/-0.2）
   - 角色：惩罚高速、大姿态角和大角速度，鼓励稳定飞行和着陆
   - 使用 next_obs 的速度、姿态角和角速度，因为这是动作执行后的结果
   - 权重适中，不会过度抑制移动，但能防止高速撞击和姿态失稳

3. **soft_landing_bonus**（任务完成 proxy，权重2.0）
   - 角色：对成功着陆的软近似奖励
   - 条件：距离目标<0.5、速度<0.3、姿态角<0.2弧度、两个支撑接触
   - 权重较小，避免 agent 过早追求接触而忽略接近过程

4. **energy_penalty**（动作代价，权重-0.1）
   - 角色：轻微惩罚使用引擎，鼓励节能
   - 主发动机惩罚0.1，姿态发动机惩罚0.05，无引擎不惩罚
   - 权重很小，避免 agent 不敢使用引擎

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确说明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- info 字典为空，没有 success 或 failure 标志可用
- 使用这些组件会诱导 LLM 发明不存在的 info 字段，违反设计原则

## 留到后续迭代的组件

- **terminal_success_reward**：当 wrapper 明确暴露 success 标志后再加入
- **terminal_failure_penalty**：当能区分失败终止后再加入
- **time_penalty**：如果 agent 能接近但拖太久，再小权重加入
- **gated_reward**：如果安全被进度奖励抵消，再加入安全门控
- **potential_based_shaping**：如果需要更标准的 shaping，再替换 progress_delta_reward

## 训练后应该观察的 failure mode

1. **goal_near_oscillation**：目标附近震荡，说明 progress_delta_reward 在目标附近信号弱
2. **high_reward_without_success**：获得高奖励但未成功着陆，说明 soft_landing_bonus 条件太宽
3. **fast_crash_near_goal**：接近目标时高速撞击，说明 stability_penalty 权重不足
4. **agent_afraid_to_move**：agent 不敢移动，说明 energy_penalty 或 stability_penalty 过强
5. **contact_reward_hacking**：agent 通过非正常方式触发接触，说明 soft_landing_bonus 条件需要收紧
