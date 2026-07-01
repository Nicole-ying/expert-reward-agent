# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 1. 主学习信号：基于水平速度的前进奖励 ==========
    # 使用 next_obs[2] (horizontal_velocity) 作为前进速度信号
    # 目标：鼓励机器人向前行走，速度越快奖励越高
    # 使用平滑的指数形式，避免梯度消失或爆炸
    temperature = 0.5  # 控制奖励曲线的陡峭程度
    forward_velocity = next_obs[2]  # 水平速度
    # 当速度为0时奖励为0，速度为正时奖励为正，速度为负时奖励为负
    forward_reward = 2.718281828 ** (forward_velocity / temperature) - 1.0
    # 限制范围避免极端值
    forward_reward = max(-10.0, min(10.0, forward_reward))
    
    # ========== 2. 稳定约束：姿态角惩罚 ==========
    # 使用 next_obs[0] (hull_angle) 惩罚身体倾斜
    # 目标：保持身体直立，避免摔倒
    hull_angle = next_obs[0]  # 主体角度，0为直立
    angle_penalty_weight = 0.5
    angle_penalty = -angle_penalty_weight * (hull_angle ** 2)
    
    # ========== 3. 稳定约束：角速度惩罚 ==========
    # 使用 next_obs[1] (hull_angular_velocity) 惩罚身体旋转
    # 目标：减少身体晃动，保持稳定
    angular_velocity = next_obs[1]
    angular_penalty_weight = 0.2
    angular_penalty = -angular_penalty_weight * (angular_velocity ** 2)
    
    # ========== 4. 效率约束：动作代价（小权重） ==========
    # 使用 action 计算能量消耗，鼓励高效行走
    energy_penalty_weight = 0.01
    energy_penalty = -energy_penalty_weight * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)
    
    # ========== 组合总奖励 ==========
    total_reward = forward_reward + angle_penalty + angular_penalty + energy_penalty
    
    # ========== 构建组件字典 ==========
    components = {
        "forward_reward": forward_reward,
        "angle_penalty": angle_penalty,
        "angular_penalty": angular_penalty,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **forward_reward**（主学习信号）：基于水平速度的指数奖励，鼓励机器人向前行走。使用指数形式使得速度越高奖励增长越快，同时负速度（后退）会得到负奖励。

2. **angle_penalty**（稳定约束）：惩罚身体倾斜角度，鼓励机器人保持直立姿态，这是稳定行走的基础。

3. **angular_penalty**（稳定约束）：惩罚身体角速度，减少身体晃动，促进平稳行走。

4. **energy_penalty**（效率约束）：小权重惩罚动作幅度，鼓励高效行走，避免不必要的能量消耗。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确说明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`，info 始终为空字典
- 没有显式的 success/failure 信号可用，因此无法可靠地实现终端奖励
- 摔倒终止由环境内部处理，我们无法在奖励函数中准确判断

## 留到后续迭代的组件

- **progress_delta_reward**：当前版本使用速度作为主信号，后续可以尝试基于位置变化的 delta 奖励
- **contact-based proxy**：当前未使用接触信号，后续可以加入步态周期相关的奖励
- **time_penalty**：如果训练发现机器人原地不动，可以加入时间惩罚
- **gated_reward**：如果稳定约束不够，可以加入安全门控机制

## 训练后应观察的 failure mode

1. **high_reward_without_success**：机器人可能学会原地快速晃动获得高速度奖励但实际不前进
2. **agent_afraid_to_move**：如果能量惩罚或稳定惩罚过重，机器人可能不敢移动
3. **unstable_gait**：虽然速度高但步态不稳定，容易摔倒
4. **contact_hacking**：如果后续加入接触奖励，可能学会不自然的接触模式
