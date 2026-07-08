# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Potential‑based shaping hyper‑parameters
    w_vel = 0.1            # weight for absolute linear velocity sum
    w_angle = 0.5          # weight for absolute body angle
    shaping_scale = 50.0   # scales the shaping signal
    gamma = 0.99           # discount factor for shaping

    # Landing continuous reward hyper‑parameters
    alpha = 5.0            # x‑position penalty (0 = target)
    beta = 5.0             # y‑position penalty
    delta = 1.0            # speed penalty exponent coefficient
    eta = 10.0             # angle penalty exponent coefficient

    # ---------- potential functions ----------
    dist_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    vel_curr = abs(obs[2]) + abs(obs[3])
    vel_next = abs(next_obs[2]) + abs(next_obs[3])

    angle_curr = abs(obs[4])
    angle_next = abs(next_obs[4])

    phi_curr = -(dist_curr + w_vel * vel_curr + w_angle * angle_curr)
    phi_next = -(dist_next + w_vel * vel_next + w_angle * angle_next)

    shaping_reward = shaping_scale * (gamma * phi_next - phi_curr)

    # ---------- landing continuous reward ----------
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    landing_reward = 0.0

    if left_contact > 0.5 and right_contact > 0.5:
        x = next_obs[0]
        y = next_obs[1]
        vx = next_obs[2]
        vy = next_obs[3]
        angle = next_obs[4]

        score = (2.718281828 ** (-alpha * x ** 2)) * \
                (2.718281828 ** (-beta * y ** 2)) * \
                (2.718281828 ** (-delta * (vx ** 2 + vy ** 2))) * \
                (2.718281828 ** (-eta * angle ** 2))
        landing_reward = score

    total_reward = shaping_reward + landing_reward

    components = {
        'shaping_reward': shaping_reward,
        'landing_reward': landing_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的组件与角色

1. **`shaping_reward` – 主学习信号**  
   - **角色**：每步都有梯度的稠密引导信号，驱动飞行器向目标靠近、减速并保持水平姿态。  
   - **数学形态**：势能基于塑形（potential-based shaping），势函数 Φ = -(距离 + w_vel·|速度| + w_angle·|角度|)，奖励为 `γ·Φ(next) - Φ(now)`，放大 50 倍。  
   - **优势**：同时关心位置、速度与姿态，避免纯距离奖励引导出高速冲撞、忽略姿态的问题。每个时间步都提供有意义的梯度，不依赖二值事件。

2. **`landing_reward` – 任务完成近似信号（proxy）**  
   - **角色**：当飞行器双足同时接触时，根据位置精度、速度、姿态的连续指数函数给予额外奖励，鼓励精准、稳定的着陆。  
   - **数学形态**：若 `left_contact` 和 `right_contact` 均为 1.0，则输出 `exp(-5x²)·exp(-5y²)·exp(-(vx²+vy²))·exp(-10·angle²)`；否则为 0。  
   - **为什么是连续而非二值**：即使双足已接触，若位置偏、速度大或倾斜，奖励仍会降低，保持梯度，避免 agent 随便蹭上平台就停止优化。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境中 **没有显式 success flag**（`explicit_success_flag_available=false`），info 为空字典，无法可靠判断成功或失败。  
- 强行发明 `info['success']` 或 `info['failure']` 会导致幻觉，违反信号可用性原则。

## 与之前尝试的结构有何不同

- 之前的 `approach_reward + soft_landing_proxy + stability_penalty` 结构中，approach 仅为距离信号，stability_penalty 作为独立组件，soft_landing_proxy 为二值条件。  
- 当前方案用 **势能塑形** 将距离、速度和姿态的期望行为统一到一个差分信号中，去掉了独立的稳定性惩罚，避免了重复信号；同时将 proxy 从二值改为连续指数形式，提供更平滑的着陆梯度。  
- 这属于 **数学机制的实质性改变**，而非仅调整系数。

## 哪些组件留到后续迭代

- 能耗 / 时间效率惩罚（`energy_penalty`、`time_penalty`）：待任务成功率达到满意水平后再引入，避免过早压制探索。  
- 更复杂的门控或课程机制：当前阶段先保证基本着陆能力。

## 训练后应观察的 failure mode

- **盘旋不落地**：shaping 信号让飞行器在目标上方低速徘徊，却迟迟不触发双足接触。应检查 landing_reward 的激活频率和强度是否不足以引导着陆。  
- **过早“假着陆”**：飞行器在平台边缘蹭触但位置偏远、速度不稳，landing_reward 仍然较小，但 shaping 可能已被部分满足。需观察 landing 得分是否随位置/速度改善明显上升。  
- **着陆后还在调整导致 episode 不终止**：若环境要求静止后自动终止，某些参数（如 delta、eta）可能偏大导致 agent 不敢完全静止。可视情况微调权重或收敛条件。
