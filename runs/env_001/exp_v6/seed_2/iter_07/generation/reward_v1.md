# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Parameters
    gamma = 0.995
    w_v = 0.02  # velocity penalty weight
    w_a = 0.1   # angle penalty weight
    w_w = 0.05  # angular velocity penalty weight
    landing_bonus = 2.0
    # Thresholds for soft landing proxy
    dist_thresh = 0.3
    speed_thresh_x = 0.1
    speed_thresh_y = 0.15
    angle_thresh = 0.15

    # 1. Potential based shaping:
    #    Phi(s) = - distance_to_target
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    phi_obs = -dist_obs
    phi_next = -dist_next
    potential_shaping = gamma * phi_next - phi_obs  # = -gamma*dist_next + dist_obs

    # 2. Stability penalty (encourages low speed, small angle, low angular velocity)
    stability_penalty = - (
        w_v * (abs(next_obs[2]) + abs(next_obs[3])) +
        w_a * abs(next_obs[4]) +
        w_w * abs(next_obs[5])
    )

    # 3. Soft landing proxy (indicates successful touchdown without explicit success flag)
    soft_landing_proxy = 0.0
    if (dist_next < dist_thresh
            and abs(next_obs[2]) < speed_thresh_x
            and abs(next_obs[3]) < speed_thresh_y
            and abs(next_obs[4]) < angle_thresh
            and next_obs[6] == 1.0
            and next_obs[7] == 1.0):
        soft_landing_proxy = landing_bonus

    total_reward = potential_shaping + stability_penalty + soft_landing_proxy

    components = {
        "potential_shaping": potential_shaping,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及其角色

1. **potential_based_shaping（主学习信号）**  
   势能函数定义为 `Phi(s) = - distance_to_target`，奖励等于 `gamma * Phi(next_obs) - Phi(obs)`，即 `dist(obs) - gamma * dist(next_obs)`。  
   - **角色**：提供密集的、可归因的引导信号，鼓励智能体每一步缩短与目标着陆垫的距离。
   - **为什么不用 `progress_delta_reward`**：前序实验中 `progress_delta` 为主的骨架已被标记为失败，因此改用势能塑形作为完全不同的主信号，数学形式上更接近标准塑形理论，且天然带有折扣因子，能缓和目标附近的震荡。

2. **stability_penalty（稳定/安全约束）**  
   - **角色**：惩罚过大的线速度、机体倾角与角速度，强迫智能体学会以平稳、低速的姿态接近目标，这对安全着陆至关重要。
   - **权重选择**：权重较小（0.02~0.1），既提供有效约束又不压制主信号的探索能力，避免 agent 不敢移动。

3. **soft_landing_proxy（任务完成近似信号）**  
   - **角色**：当智能体同时满足“靠近目标、低速、姿态水平、双支撑脚接触”时给予一次固定奖励，提供一个到达并稳定落地的微弱正向信号。由于环境中没有显式 success flag，这是唯一接近任务完成的奖励。
   - **设计谨慎**：多个阈值和接触标志共同使用，避免因接触标志单独为真就错误奖励，防止 reward hacking。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- **环境卡片明确标注 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`**，且 `info` 为空字典。任何对 `info['success']` 或终止原因的假设都会导致奖励函数失效或幻觉。因此 v1 完全不依赖终点显式信号，所有学习信号仅从 `obs` 和 `next_obs` 提取。

## 留到后续迭代的组件

- **energy_penalty / engine_use_cost**：目前智能体还不会飞向目标，过早惩罚动作使用会导致 agent 不敢点火，造成 `agent_afraid_to_move`。待基本接近策略稳定后再加入。
- **time_penalty**：同理，避免智能体为减少步数而采取冒险行为，在基本可达后再优化效率。
- **gated_reward**：更复杂的安全门控或阶段切换逻辑，待基础稳定后用于细化碰撞防护或着陆段专门奖励。
- **terminal_success/failure rewards**：若后续环境 wrapper 明确暴露成功/失败标志时，可以替换或加强 soft_landing_proxy。

## 训练后应重点观察的 failure mode

- **目标附近震荡（goal_near_oscillation）**：势能塑形在距离很小时可能给出很小甚至负的奖励，结合稳定性惩罚可能导致智能体在目标附近来回徘徊而不降速；需检查 `dist` 和 `speed` 的稳态分布。
- **高奖励但未成功（high_reward_without_success）**：soft_landing_proxy 的阈值可能被部分满足（如 slow 但 contact=0）仍获得较高总奖励，但实际上未着陆；需观察着陆成功的比例曲线。
- **高速撞击目标附近**：如果 stability_penalty 权重过小，智能体可能学会快速冲向目标后刹不住，触发 crash 或飞出视口；应监控终止类型分布及碰撞频率。
- **姿态失控**：角速度或角度惩罚若被主信号压倒，智能体可能在接近过程中翻滚，影响双支撑脚触地；需跟踪角度与角速度的时间序列。