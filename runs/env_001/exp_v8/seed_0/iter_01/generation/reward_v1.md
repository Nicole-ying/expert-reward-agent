# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置为 (0,0)，平台中心
    cx, cy = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]

    # 当前位置与目标距离
    d_curr = (cx ** 2 + cy ** 2) ** 0.5
    # 下一时刻与目标距离
    d_next = (nx ** 2 + ny ** 2) ** 0.5

    # 1. 主学习信号：逐步靠近目标
    progress_reward = d_curr - d_next

    # 2. 轻量稳定约束，基于下一状态
    vx, vy = next_obs[2], next_obs[3]
    speed = (vx ** 2 + vy ** 2) ** 0.5
    angle = abs(next_obs[4])
    angular_v = abs(next_obs[5])

    stability_penalty = (
        -0.05 * speed
        - 0.1 * angle
        - 0.05 * angular_v
    )

    # 3. 软着陆完成近似信号
    soft_landing_proxy = 0.0
    if (
        d_next < 0.15
        and speed < 0.2
        and angle < 0.1
        and next_obs[6] == 1.0
        and next_obs[7] == 1.0
    ):
        soft_landing_proxy = 1.0

    # 总奖励
    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward,
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **progress_reward（主学习信号）**：  
  每一步更靠近目标平台中心（即 `obs[0]` 和 `obs[1]` 更接近 0）时获得正向奖励。它为 agent 提供了每步可用的密集梯度，是奖励函数中贡献最大的正向部分。

- **stability_penalty（轻量稳定约束）**：  
  基于下一状态的速度、机体倾角和角速度施加线性惩罚，鼓励 agent 低速、垂直、无明显旋转。权重均较小，避免压制探索，但足以抑制粗暴运动。

- **soft_landing_proxy（任务完成近似信号）**：  
  当 agent 同时满足“极靠近目标”、“低速”、“近乎竖直”、“两条支撑腿均接触平台”时，给予一个固定奖励。它不是真正的成功标志，而是用多个连续条件组合的 soft proxy，避免单一接触信号被 exploit。

- **为什么未使用 terminal_success_reward / terminal_failure_penalty**：  
  环境未提供显式成功/失败标志，`info` 为空；`terminal` 信号无法区分成功与失败终止，故不适合作 v1 核心项。

- **后续迭代预留**：  
  效率类组件（`energy_penalty` 对引擎使用惩罚、`time_penalty` 对步数惩罚）在 agent 能稳定完成任务后加入；若出现目标附近高频震荡，可考虑 `progress_reward` 作 clip 或引入 `gated_reward` 进行安全门控。

- **训练后应重点观察的 failure mode**：  
  1. **目标附近来回摆动**：`progress_reward` 可能鼓励 agent 在目标区域前后移动以持续获得正向差奖励，尽管 `stability_penalty` 会抑制高速，但仍需观察。  
  2. **提前坠毁或高速触地**：`soft_landing_proxy` 的阈值较严格，若 agent 在触发前就因为速度过大或角度失控而 crash，则奖励主要依赖 progress + stability 不足以阻止。可能需要调大稳定惩罚或增加安全区域信号。  
  3. **rare proxy 触发导致学习停滞**：若 agent 难以同时满足所有 soft landing 条件，则 `soft_landing_proxy` 长期为 0，可能对学习帮助有限；后续可考虑放宽阈值或改用 gradual bonus（如基于距离和速度的连续函数）。