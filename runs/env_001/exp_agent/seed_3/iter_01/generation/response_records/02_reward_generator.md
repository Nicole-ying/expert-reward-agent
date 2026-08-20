# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== Helper: distance to goal ==========
    # Goal is assumed to be at (x=0, y=0) relative to the landing pad centre / reference height
    def dist_to_goal(x, y):
        return (x**2 + y**2) ** 0.5

    # ========== 1. Progress delta reward (main learning signal) ==========
    d_current = dist_to_goal(obs[0], obs[1])
    d_next = dist_to_goal(next_obs[0], next_obs[1])
    progress_delta = d_current - d_next   # positive when getting closer

    # ========== 2. Stability penalty (light constraint) ==========
    # Penalise high velocity, large tilt and fast rotation
    w_vel   = 0.1
    w_angle = 0.5
    w_omega = 0.1

    speed_penalty = w_vel * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    omega_penalty = w_omega * abs(next_obs[5])

    stability_penalty = -(speed_penalty + angle_penalty + omega_penalty)

    # ========== 3. Soft landing proxy (task‑completion hint) ==========
    # A small bonus when all conditions are met: near target, slow, upright and both supports in contact.
    near_target   = d_next < 0.2
    low_speed     = (abs(next_obs[2]) < 0.2) and (abs(next_obs[3]) < 0.2)
    stable_angle  = abs(next_obs[4]) < 0.1
    both_contact  = (next_obs[6] == 1.0) and (next_obs[7] == 1.0)

    soft_landing_proxy = 0.2 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # ========== Total reward ==========
    # Scale progress_delta to a meaningful magnitude (environment‑dependent, tune later)
    total_reward = 10.0 * progress_delta + stability_penalty + soft_landing_proxy

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及其角色

1. **progress_delta_reward（主学习信号）**  
   - 计算方式：`d(obs, goal) − d(next_obs, goal)`，其中 `goal = (0,0)`。  
   - 角色：密集过程引导，每一步鼓励 agent 减小与目标的距离，是导航任务最直接的学习信号。  
   - 权重设为 `10.0`，以便在假定距离变化较小时仍能提供足够梯度的主导信号。

2. **stability_penalty（稳定与安全约束）**  
   - 包含三项：线速度惩罚、机体角度惩罚、角速度惩罚。  
   - 角色：轻量约束，防止高速撞击、过度倾斜或快速旋转，促使 agent 学会平稳接近与降落。  
   - 使用较小的权重，避免压制主信号。

3. **soft_landing_proxy（任务完成近似）**  
   - 当同时满足“足够近”、“速度低”、“姿态正”、“双脚触地”四个条件时，给予一个小额固定奖励。  
   - 角色：提供触地阶段的方向性提示，帮助 agent 过渡到最终静止状态，但不伪造 success flag。  
   - 权重很小（0.2），防止 agent 围绕条件“刷分”而不真正完成任务。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡片明确指出 `explicit_success_flag_available: false` 且 `explicit_failure_flag_available: false`，`info` 为空字典。  
- 若强行使用，将迫使奖励函数发明不存在的 `info['success']` 或误判终止原因，导致不可靠的学习信号。  
- 本版改为依靠**过程奖励 + 轻量 proxy** 引导行为，成功/失败的区分留到后续 wrapper 提供明确信号后再引入。

## 哪些组件留到后续迭代

- **energy_penalty / fuel penalty**：过早加入可能使 agent 不敢点火，导致“不动”的问题。待 agent 能稳定到达目标附近后再加入以优化能耗。  
- **time_penalty**：每步固定代价。易导致冒险快速失败，通常待接近率达一定水平后才添加。  
- **gated_reward / curriculum**：更复杂的多阶段或安全门控逻辑，适合在基础策略收敛后引入。  
- **terminal_success_reward / terminal_failure_penalty**：一旦 wrapper 明确暴露成功/失败标志，可显著提升最终任务完成率。  

## 训练后应观察的 failure mode

1. **goal_near_oscillation**：agent 在目标附近来回摆动，无法稳定着陆。  
   - 可能原因：progress_delta 引导过强且缺乏阻尼，需要调大稳定性惩罚或裁剪 progress_delta。  
2. **high_reward_without_success**：奖励很高但并未真正停在垫上。  
   - 可能原因：soft_landing_proxy 条件过松或接触奖励被利用；应收紧阈值或降低 proxy 权重。  
3. **fast_crash_near_goal**：沿 progress 方向快速冲向目标但姿态失控或侧翻。  
   - 需检查稳定性惩罚是否足够，或需提升姿态/角速度权重。  
4. **contact reward hacking**：agent 刻意制造双脚接触但不靠近目标。  
   - 由于 soft_landing_proxy 同时要求“近目标”，可缓解此问题；若仍出现，可提高近目标阈值或直接移除接触条件。
