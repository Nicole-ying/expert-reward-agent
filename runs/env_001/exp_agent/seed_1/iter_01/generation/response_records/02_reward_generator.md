# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward - 奖励每一步更接近目标
    # 计算当前位置到目标(0,0)的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta  # 权重10，鼓励持续接近
    
    # 稳定约束：stability_penalty - 惩罚高速、大姿态角和角速度
    # 使用next_obs的状态，因为动作影响后的状态更相关
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚：平方形式，对高速更敏感
    speed_penalty = -0.5 * (x_vel ** 2 + y_vel ** 2)
    # 姿态角惩罚：角度越大惩罚越大，使用平方
    angle_penalty = -0.3 * (body_angle ** 2)
    # 角速度惩罚：旋转越快惩罚越大
    angular_penalty = -0.2 * (angular_vel ** 2)
    
    stability_penalty = speed_penalty + angle_penalty + angular_penalty
    
    # 任务完成proxy：soft_landing_proxy - 当接近目标且稳定时给予小奖励
    # 条件：距离<0.5，速度<0.3，角度<0.2，角速度<0.2，且至少一个支撑接触
    near_target = next_dist < 0.5
    low_speed = (x_vel ** 2 + y_vel ** 2) ** 0.5 < 0.3
    stable_angle = abs(body_angle) < 0.2
    low_angular_vel = abs(angular_vel) < 0.2
    has_contact = (next_obs[6] > 0.5) or (next_obs[7] > 0.5)
    
    if near_target and low_speed and stable_angle and low_angular_vel and has_contact:
        soft_landing_bonus = 2.0
    else:
        soft_landing_bonus = 0.0
    
    # 动作代价：energy_penalty - 小权重惩罚使用引擎
    # 动作1、2、3都使用引擎，动作0不使用
    if action == 0:
        energy_penalty = 0.0
    else:
        energy_penalty = -0.1  # 小权重，避免agent不敢动
    
    # 总奖励
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    # 组件字典
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

1. **progress_delta_reward**（主学习信号，权重10）
   - 角色：引导智能体持续接近目标位置
   - 计算方式：当前距离减去下一时刻距离，正值表示更接近目标
   - 这是最直接的任务引导信号，让智能体学会向目标移动

2. **stability_penalty**（稳定约束，负向）
   - 角色：惩罚高速、大姿态角和角速度，促进稳定飞行
   - 包含三个子项：速度惩罚（-0.5*速度平方）、姿态角惩罚（-0.3*角度平方）、角速度惩罚（-0.2*角速度平方）
   - 使用平方形式使惩罚对较大偏差更敏感

3. **soft_landing_bonus**（任务完成proxy，+2.0）
   - 角色：当智能体接近目标且稳定时给予额外奖励
   - 条件：距离<0.5、速度<0.3、角度<0.2、角速度<0.2、至少一个支撑接触
   - 这是对成功着陆的软近似，但不等于真正的success flag

4. **energy_penalty**（动作代价，-0.1）
   - 角色：轻微惩罚使用引擎，鼓励节能
   - 权重很小（-0.1），避免智能体不敢移动

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确声明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- info字典始终为空，没有可用的成功/失败标志
- 使用这些项会诱导伪造信号或误判终止原因

## 留到后续迭代的组件

- **terminal_success_reward**：当wrapper明确暴露success flag后再加入
- **terminal_failure_penalty**：当能区分失败终止后再加入
- **time_penalty**：如果智能体能接近目标但拖太久，后续小权重加入
- **gated_reward**：如果安全被进度奖励抵消，后续再加入安全门控
- **potential_based_shaping**：如果需要更标准的塑形，后续替换或补充progress_delta

## 训练后应观察的failure mode

1. **goal_near_oscillation**：目标附近震荡，不完成着陆
   - 观察：progress_delta_reward接近0但soft_landing_bonus不触发
   - 对策：增大soft_landing_bonus权重或收紧条件

2. **high_reward_without_success**：获得高奖励但未成功着陆
   - 观察：progress_delta_reward高但soft_landing_bonus不触发
   - 对策：收紧soft_landing_bonus条件或加入时间惩罚

3. **fast_crash_near_goal**：接近目标时高速撞击
   - 观察：progress_delta_reward高但stability_penalty也高
   - 对策：增大stability_penalty权重或加入速度上限惩罚

4. **agent_afraid_to_move**：不敢使用引擎，停留在起点
   - 观察：所有奖励接近0，energy_penalty主导
   - 对策：降低energy_penalty权重或移除
