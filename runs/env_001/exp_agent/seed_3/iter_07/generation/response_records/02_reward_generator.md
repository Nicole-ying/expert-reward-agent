# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主信号：到原点的负距离（目标位置为 (0,0)）
    dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    distance_reward = -1.0 * dist

    # 稳定约束：惩罚速度、倾斜角和角速度的绝对值
    vel_penalty = 0.1 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.2 * abs(next_obs[4])
    angvel_penalty = 0.05 * abs(next_obs[5])
    stability_penalty = -vel_penalty - angle_penalty - angvel_penalty

    # 软着陆代理：当满足安全着陆条件时给予一次性正奖励
    landing_bonus = 0.0
    if (dist < 0.1 and 
        abs(next_obs[2]) < 0.2 and abs(next_obs[3]) < 0.2 and 
        abs(next_obs[4]) < 0.05 and abs(next_obs[5]) < 0.1 and 
        next_obs[6] == 1.0 and next_obs[7] == 1.0):
        landing_bonus = 5.0  # 一次性的正奖励

    total_reward = distance_reward + stability_penalty + landing_bonus

    components = {
        'distance_reward': distance_reward,
        'stability_penalty': stability_penalty,
        'landing_bonus': landing_bonus,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 奖励组件清单与角色

1. **distance_reward** – 主学习信号  
   - 计算飞行器到目标平台中心的欧氏距离，奖励为 `-1.0 * distance`。  
   - 角色：提供密集的下降‑趋近引导，让 agent 学会向原点移动。  
   - 注意：与前序失败的 `progress_delta_reward` 不同，这里直接用距离绝对值，不依赖两步差分，避免震荡和 shape 误导。

2. **stability_penalty** – 稳定约束（安全/风格约束）  
   - 对水平速度、垂直速度、机体倾斜角度和角速度施加小权重惩罚。  
   - 角色：抑制高速飞行、剧烈摆动和大幅倾斜，促使 agent 在接近目标时自觉减速并保持竖直。  
   - 权重较小（速度 0.1，角度 0.2，角速度 0.05），避免过度限制探索。

3. **landing_bonus** – 任务完成软代理  
   - 当飞行器位置很近（<0.1）、速度低、角度小、角速度低且双腿同时接触时，给一次性 +5 奖励。  
   - 角色：在无显式成功标记时，以启发式条件提供“正确完成”的稀疏正向反馈。  
   - 条件严格（距离 <0.1，速度 <0.2，角度 <0.05，角速度 <0.1），防止 agent 单纯通过接触信号刷分。

## 与本轮约束的对应

- **为什么没用 `terminal_success_reward` 或 `terminal_failure_penalty`**  
  信息卡明确 `explicit_success_flag_available=false`，info 为空，没有可靠的完成或失败标志；故不加入任何依赖终止信号的奖励项。

- **为什么没加入 energy_penalty、time_penalty 等**  
  这些属于效率/动作成本约束，通常在学习后期优化燃料或时间时引入；v1 优先解决“能否稳定着陆”，避免 agent 因为吝惜引擎而不敢移动。

- **distance_reward 与 progress_delta 的区别**  
  前序失败的骨架采用 `progress_delta_reward`（距离差）作为主信号，本轮改用直接的负距离，提供一个单调、平稳的梯度，且不涉及差分带来的高方差和错误累积。

## 后续迭代计划

- **energy_penalty / action_cost**：当 agent 能稳定着陆后，再引入对使用引擎的小幅惩罚，以追求燃料效率。
- **terminal_success_reward**：如果后续 wrapper 提供了显式 success flag，可替换或强化 `landing_bonus`。
- **gated_reward 或阶段课程**：若发现 agent 始终在目标附近徘徊却不接触，可考虑分阶段引入接触要求。
- **阈值与权重调优**：根据早期训练曲线和回放分析，调整 `stability_penalty` 权重和 `landing_bonus` 的阈值。

## 预期观察的 failure mode

- **距离最小化但不停靠**：agent 可能学会悬停在目标上方一小段距离，获得高 `distance_reward`，但不触发 `landing_bonus`；此时需要检查速度/姿态约束是否过强，或适当放宽着陆条件。
- **过早接触式投机**：若 `landing_bonus` 条件设置过宽，agent 可能高速撞击双腿触地骗取奖励，此时应减小接触相关的阈值或增大速度惩罚。
- **角速度/姿态惩罚过强导致的保守行为**：agent 始终维持低速、细微姿态，迟迟不敢下降，需适当减小 `stability_penalty` 权重。
- **双腿接触不对称**：agent 可能以单腿支撑获得稳定，若长期不触发 `landing_bonus`，可考虑在后续迭代中加入对单腿接触的轻微惩罚。
