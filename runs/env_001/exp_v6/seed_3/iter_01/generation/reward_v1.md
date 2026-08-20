# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- main progress signal: reduction in Euclidean distance to target ----
    # target position is (0, 0) in the relative coordinate system
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5
    progress_delta = dist_obs - dist_next   # positive when moving closer

    # ---- light stability penalty on next observation ----
    # penalise linear velocity, body angle and angular velocity
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # weights chosen to be small so they don't dominate progress
    w_vel = 0.01
    w_angle = 0.01
    w_angvel = 0.005

    stability_penalty = (
        - w_vel * (abs(vx) + abs(vy))
        - w_angle * abs(angle)
        - w_angvel * abs(ang_vel)
    )

    # ---- total reward ----
    total_reward = progress_delta + stability_penalty

    # ---- component logging ----
    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及角色

1. **progress_delta_reward**（主学习信号）  
   - 角色：密集过程引导，告诉智能体每一步是否更靠近目标着陆台。  
   - 计算方式：当前距离 `dist_obs` 与下一时刻距离 `dist_next` 的差值，正值表示靠近目标，负值表示远离。  
   - 距离使用二维欧氏距离 `sqrt((x-0)^2 + (y-0)^2)`，因为目标台中心位于相对坐标 `(0,0)`。

2. **stability_penalty**（轻量稳定约束）  
   - 角色：鼓励平稳接近与软着陆，抑制高速、大倾角、剧烈旋转。  
   - 包含三个子项：线速度绝对值惩罚（`vx, vy`）、主体角度绝对值惩罚（`angle`）、角速度绝对值惩罚（`ang_vel`），均作用在 `next_obs` 上。  
   - 权重设置很小（`w_vel=0.01`，`w_angle=0.01`，`w_angvel=0.005`），避免过度限制探索或导致“不敢动”的现象。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- 环境卡片明确标记 `explicit_success_flag_available: false` 且 `explicit_failure_flag_available: false`，`info` 字典为空。  
- 依专家知识约束，不能在奖励函数中发明或推测 `info['success']` 或 `info['failure']`，因此终点成功/失败奖励不作为 v1 核心项。  
- 未来若环境 wrapper 显式提供成功/失败标志，可在后续迭代中加入 `terminal_success_reward` 与 `terminal_failure_penalty`。

## 哪些组件留到后续迭代
- **soft_landing_proxy**：腿接触 + 低速度 + 小角度 + 近目标的小奖励，当前未加，避免接触信号过早影响学习重心；若 v1 出现“到达目标附近但不着陆”的 failure mode，再引入并收紧条件。  
- **energy_penalty / time_penalty**：动作代价或每步固定代价，v1 暂不加入，以防止智能体不敢点火或冒险快速失败；当靠近与稳定基础形成后，再通过小权重鼓励节能。  
- **gated_reward / dynamic curriculum**：更复杂的门控或分阶段奖励，目前不需在 v1 中引入，以免增加调试难度。

## 训练后应重点观察的 failure mode

- **high_reward_without_success**：距离奖励持续有效，但智能体未能真正软着陆（例如在目标附近悬停但不接触、或单腿接触后无法稳定）。  
- **fast_crash_near_goal**：接近过程中速度未有效降低，导致高速撞击着陆台或硬接触。  
- **oscillation / 抖动**：在目标附近由于 `progress_delta` 不施加最终停止的激励，可能出现反复徘徊。  
- **过强的稳定性约束导致畏惧运动**：若权重过大，智能体可能选择不点火、直接坠落，牺牲快速到达性。可在训练早期观察主发动机使用频率判断。  

观察上述模式有助于在下一版奖励迭代中针对性调整（例如增大稳定性惩罚权重、加入 soft landing proxy 等）。