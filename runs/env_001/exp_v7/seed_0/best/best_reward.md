# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ======== 提取观测 ========
    x_t, y_t = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    w_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ======== 1. 主学习信号：进步奖励 (progress_delta) ========
    dist_t = (x_t ** 2 + y_t ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = dist_t - dist_next   # 正值为接近目标
    progress_reward = 0.5 * progress   # 缩放因子，使单步典型值在 0.0~0.1 左右

    # ======== 2. 软着陆近似奖励 (soft_landing_proxy) ========
    # 条件：接近目标、低速、角度小、双腿都已接触
    near_target = dist_next < 0.1
    low_speed = (abs(vx_next) + abs(vy_next)) < 0.1
    stable_angle = abs(angle_next) < 0.1
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_reward = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_reward = 1.0   # 单步事件型奖励，鼓励稳定着陆

    # ======== 3. 轻量稳定性惩罚 (stability_penalty) ========
    # 小权重抑制高速、大角度、高角速度，但不压制主学习信号
    stability_penalty = -0.002 * (abs(vx_next) + abs(vy_next)) \
                        - 0.002 * abs(angle_next) \
                        - 0.001 * abs(w_next)

    # ======== 总奖励 ========
    total_reward = progress_reward + soft_landing_reward + stability_penalty

    # ======== 组件字典 ========
    components = {
        'progress_reward': progress_reward,
        'soft_landing_reward': soft_landing_reward,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **使用组件**：
  1. `progress_reward` (主学习信号)：每一步测量到目标点的欧氏距离变化，鼓励 agent 朝着目标垫方向移动。选用 `progress_delta` 骨架而非 `distance_reward`，避免与后续可能加入的其他距离相关信号产生重复梯度，同时提供正向的接近信号。
  2. `soft_landing_reward` (任务完成近似信号)：当飞行器靠近目标、速度极低、姿态水平、且双腿均接触地面时，给予一次较大的事件奖励。这是一个多条件组合的 proxy，用于在缺少显式 success flag 时引导 agent 完成平稳着陆。条件设置较严格（距离 <0.1，绝对速度之和 <0.1，角度 <0.1 rad，双腿同时接触），以降低被 exploit 的风险（如轻触地面弹起）。
  3. `stability_penalty` (轻量稳定约束)：对高速、大角度和高角速度施加极小惩罚，帮助抑制剧烈震荡或失控，但不让 agent 害怕动作。权重设置为 0.002~0.001，远小于主要学习信号，充当稳定的“方向盘”。

- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**：  
  环境明确声明 `explicit_success_flag_available = false` 且 `explicit_failure_flag_available = false`，info 字段为空，无法安全地引入终端成功或失败的离散奖励。强行依赖未定义字段会导致 LLM 发明不存在的信号，违反设计约束。

- **后续迭代预留**：
  - `energy_penalty` / `time_penalty`：当前阶段不加，因为有信心 agent 应先学会完成任务，再优化燃料效率。
  - 如果 `soft_landing_reward` 被 exploit（如频繁轻触地面刷分），后续可在 wrapper 中增加状态记忆或进一步收紧条件，或改为仅首次触发奖励。
  - 如果出现 goal‑near oscillation 或 high‑reward‑without‑success，后续可尝试替换或补充 `gated_reward` 或更精细的 touchdown‑success 检测。

- **需观察的 failure mode**：
  - 飞行器长时间在目标上空来回摆动，仅依靠 progress 奖励无法稳定着陆（`high_reward_without_success`）。
  - `soft_landing_reward` 触发频率过高或过低：过高可能被 exploit，过低则对学习帮助不大。
  - 稳定性惩罚过轻导致姿态失控，或过重导致不敢动作（`agent_afraid_to_move`）。
  - 接触判断（浮点值）的阈值（0.5）是否合理，需根据实际接触信号的数值范围调优。