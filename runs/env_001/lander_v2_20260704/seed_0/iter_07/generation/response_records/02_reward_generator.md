# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取 next_obs 的各个分量
    x_pos = next_obs[0]          # 水平位置
    y_pos = next_obs[1]          # 垂直位置
    x_vel = next_obs[2]          # 水平速度
    y_vel = next_obs[3]          # 垂直速度
    body_angle = next_obs[4]     # 身体角度
    left_contact = next_obs[6]   # 左支撑腿接触
    right_contact = next_obs[7]  # 右支撑腿接触

    # 主学习信号：非线性接近奖励，鼓励向目标靠近
    dist_to_target = (x_pos ** 2 + y_pos ** 2) ** 0.5
    proximity_reward = 1.0 / (1.0 + dist_to_target)

    # 稳定约束：抑制高速，防止剧烈运动
    velocity_penalty = -0.1 * (abs(x_vel) + abs(y_vel))

    # 姿态约束：抑制大幅倾斜，保证着陆姿态
    angle_penalty = -0.05 * abs(body_angle)

    # 任务完成近似信号：鼓励双腿接触着陆垫
    contact_reward = 0.3 * (left_contact + right_contact)

    total_reward = proximity_reward + velocity_penalty + angle_penalty + contact_reward

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'angle_penalty': angle_penalty,
        'contact_reward': contact_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件
- **proximity_reward**：主学习信号，使用平滑非线性函数 `1/(1+distance)` 将对目标的接近程度转化为正奖励，提供每一步稠密的梯度，驱动智能体持续靠近着陆垫。
- **velocity_penalty**：轻量稳定约束，按水平与垂直速度的绝对值之和施加惩罚，抑制剧烈运动，促使飞行器在接近目标时自动减速。
- **angle_penalty**：姿态约束，对身体倾角绝对值施加惩罚，引导智能体保持接近水平的姿态，为安全着陆创造条件。
- **contact_reward**：任务完成近似信号，对双腿每一条的接触标志给予独立正奖励，鼓励着陆腿与垫面接触，是迈向成功的最后一步。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境卡片明确声明 `explicit_success_flag_available: false` 且 `explicit_failure_flag_available: false`，info 字典为空，无法访问任何成功或失败标志。因此 v1 设计中不使用任何需要显式终局信号的组件，也不尝试伪造 success/failure。

## 与之前尝试的结构差异
此前尝试的结构（如 `distance_reward + stability_penalty + soft_landing_proxy`）多采用线性负距离奖励或状态型组合式着陆代理信号，容易导致 agent 在目标附近悬停获取高频奖励（high_reward_without_success）。本设计做出以下关键改变：
- **将距离信号从线性负值改为非线性正奖励**：`1/(1+dist)` 数学形态提供靠近时急剧上升的奖励曲线，同时避免恒定负值对探索的压制，并使奖励值整体为正，减小与惩罚项的尺度冲突。
- **放弃组合式着陆信号，改用独立接触奖励**：`contact_reward` 仅基于腿部接触标志，不与速度、角度等条件耦合，杜绝了组合条件严苛导致的稀疏化，使接触奖励直接可作为稠密梯度的一部分，同时配合稳定约束自然抑制不安全接触行为。

## 留待后续迭代的组件
- 效率类（如 `energy_penalty`、`time_penalty`）留待智能体能稳定完成任务后再加入，避免过早压制探索。
- 更复杂的门控或课程学习（如 `gated_reward`、动态 curriculum）待基础策略收敛后再考虑，以防止过早引入条件造成学习停滞。

## 训练后应观察的 failure mode
1. **目标附近悬停**：虽然采用了非线性接近奖励，但 agent 仍可能在 0 距离附近获得较高 `proximity_reward` 而不完成着陆。需观察 `contact_reward` 是否足够驱动接触行为；若出现大量高奖励无接触的 episode，可考虑降低 `proximity_reward` 系数或提高 `contact_reward`。
2. **反复轻触垫面**：如果 agent 为了频繁获取 `contact_reward` 而反复点触垫面后迅速离开，可能表明 `velocity_penalty` 或 `angle_penalty` 约束不足。此时可微调惩罚系数（增大速度约束）或在后续迭代中引入连续接触的长时间奖励。
3. **姿态失稳导致 crash**：若 `angle_penalty` 权重过低，agent 可能在俯冲时大幅倾斜，引发身体非支撑部分触地。需监控终止原因分布，若 crash 终止比例过高，应适当提高 `angle_penalty` 或增加角速度惩罚。
