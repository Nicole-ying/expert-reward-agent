# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract next observations
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Main learning signal: negative distance to landing target (dense progress guiding)
    dist_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -dist_to_target

    # 2. Stability constraint: discourage high speed, tilt, and rotation (light penalty)
    stab_weight = 0.01
    stability_penalty = -stab_weight * (abs(x_vel) + abs(y_vel) + abs(angle) + abs(ang_vel))

    # 3. Soft landing proxy: bonus when near target, low speed, upright, and both legs contact
    near_target = dist_to_target < 0.2
    low_speed = abs(x_vel) < 0.2 and abs(y_vel) < 0.2
    stable_angle = abs(angle) < 0.1
    low_ang_vel = abs(ang_vel) < 0.1
    both_legs = (left_contact == 1.0) and (right_contact == 1.0)

    if near_target and low_speed and stable_angle and low_ang_vel and both_legs:
        soft_landing_proxy = 1.0
    else:
        soft_landing_proxy = 0.0

    total_reward = distance_reward + stability_penalty + soft_landing_proxy

    components = {
        "distance_reward": distance_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及角色

1. **distance_reward（主学习信号）**  
   - 形式：`-sqrt(x² + y²)`，其中坐标相对于目标平台中心。  
   - 角色：提供每步稠密梯度，引导飞行器靠近目标平台。  
   - 优点：连续、无稀疏触发，与任务目标直接对应。  

2. **stability_penalty（轻量稳定约束）**  
   - 形式：按相同小权重惩罚线速度绝对值、倾角绝对值和角速度绝对值。  
   - 角色：抑制过大速度、倾斜和旋转，帮助维持稳定姿态和平滑运动。  
   - 权重极低（0.01），避免阻碍探索或让 agent 不敢行动。  

3. **soft_landing_proxy（任务完成近似信号）**  
   - 条件组合：距离 < 0.2、线速度均 < 0.2、倾角 < 0.1、角速度 < 0.1、双腿同时触地。  
   - 角色：多条件联合判断，近似成功着陆，在 agent 达成稳定着陆时给予正奖励。  
   - 不是单一二值接触奖赏，结合了位置、速度和姿态，降低了奖励被 hack 的风险。  

## 未使用 terminal_success_reward / terminal_failure_penalty 的原因

- 环境明确声明 `explicit_success_flag_available: false` 且 info 为空，没有任何 success/failure 标记。  
- 不可发明未提供的 info 字段，因此无法使用终端成功/失败奖励。  
- 将着陆成功信号通过观测组合实现为 soft proxy，符合约束并提供了可塑形的正向指引。  

## 后续迭代预留

- **terminal_success_reward / terminal_failure_penalty**：当环境 wrapper 明确暴露成功或失败标志后再加入。  
- **energy_penalty / time_penalty**：作为效率/动作代价，待 agent 能稳定完成任务后加入 v2 优化步数和引擎使用。  
- **gated_reward / dynamic curriculum**：复杂门控或课程学习，留到更后期迭代。  

## 预期可能的 failure mode 与观察重点

- **着陆前震荡**：distance_reward 可能使 agent 在目标上方反复摆动而不着地。需观察是否长时间停在 dist < 0.2 却未触地，必要时调整 proxy 阈值或增加速度限制。  
- **soft proxy 触地即止**：如果 agent 偶然双腿接触但速度/角度仍不满足条件，则不会获得高频 bonus，降低 hack 可能。训练初期需观察 proxy 触发频率是否极低，可适当放宽阈值。  
- **奖励尺度失衡**：当 dist 较大时，distance_reward 的负值可能远超 proxy +1，不会掩盖目标；当接近目标时，distance_reward 接近 0，proxy 将成为主要正向信号，鼓励完成着陆。需监控 total_reward 的均值及 proxy 触发率以确保平衡。