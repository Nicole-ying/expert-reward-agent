# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取下一步状态
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 权重和阈值（v1 固定值，后续迭代可调）
    w_dist = 1.0
    w_vel = 0.1
    w_angle = 0.1
    w_angvel = 0.05
    w_bonus = 20.0

    thresh_dist = 0.3
    thresh_vx = 0.2
    thresh_vy = 0.2
    thresh_angle = 0.1

    # 组件1: 距离成本（推动 agent 靠近目标着陆垫）
    distance_cost = -w_dist * (nx ** 2 + ny ** 2)

    # 组件2: 稳定性成本（抑制高速、大角度和高角速度，鼓励平稳接近）
    stability_cost = -(
        w_vel * (vx ** 2 + vy ** 2)
        + w_angle * abs(angle)
        + w_angvel * abs(omega)
    )

    # 组件3: 软着陆完成近似奖励（仅在多条件齐备时给予一次正向激励）
    dist = (nx ** 2 + ny ** 2) ** 0.5
    soft_landing_condition = (
        dist < thresh_dist
        and abs(vx) < thresh_vx
        and abs(vy) < thresh_vy
        and abs(angle) < thresh_angle
        and left_contact > 0.5
        and right_contact > 0.5
    )
    soft_landing_bonus = w_bonus if soft_landing_condition else 0.0

    # 合成总奖励
    total_reward = distance_cost + stability_cost + soft_landing_bonus

    # 组件字典（只包含直接参与 total_reward 的项）
    components = {
        'distance_cost': distance_cost,
        'stability_cost': stability_cost,
        'soft_landing_bonus': soft_landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **distance_cost（主学习信号）**  
   — 基于下一步位置的平方距离负值。该组件每步提供稠密梯度，直接引导 agent 不断靠近目标着陆垫（坐标原点）。平方形式使远端梯度更大，有利于早期探索；靠近目标后梯度平缓，避免在垫上反复震荡。

2. **stability_cost（稳定/平滑约束）**  
   — 同时惩罚下一步的线速度平方、绝对姿态角和绝对角速度。该组件扮演“方向盘”角色，防止 agent 以高速或大角度姿态冲向目标，迫使它学会减速并保持水平，为安全着陆创造条件。权重轻于主信号，避免过度保守。

3. **soft_landing_bonus（任务完成近似信号）**  
   — 当下一步同时满足“位置极近、速度极低、角度极小、双腿接触”时，给予一次性正奖励。它不是显式成功标志，而是通过多条件组合给出密集的目标完成反馈，帮助 agent 在接近目标后完成“接触并稳定”这一最后步骤。条件严格（低速度、小角度、双腿接触），降低 reward hacking 风险。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡片中 `explicit_success_flag_available = false`，`explicit_failure_flag_available = false`，且 `info` 为空，不提供任何 success、failure 或 termination_reason 字段。  
- 因此无法依赖显式标志给出终局奖励/惩罚。`soft_landing_bonus` 通过观测信号近似完成状态，不依赖于不可用的 info 字段。

## 留到后续迭代的组件

- **动作/能耗成本（energy_penalty）**：任务要求“尽量少使用引擎推力”，但过早引入会抑制探索，v1 先让 agent 学会安全着陆，后续再加入轻量能耗惩罚优化引擎使用。  
- **时间效率代价（time_penalty）**：同 energy_penalty，v1 阶段不追求步数效率。  
- **更复杂的门控或课程学习（gated_reward / dynamic_curriculum）**：当观察到 agent 在接近目标时仍 crash 或震荡，再考虑引入阶段分界。

## 训练后应关注的 failure mode

- **目标附近高速撞击**：可能 stability_cost 权重或阈值不当，导致 agent 为了减少距离成本而保持高速，未能减速触地。  
- **目标附近悬停而不接触**：soft_landing_bonus 条件过严或权重不足，agent 停在垫子上方但双腿不触地，奖励无法区分。  
- **反复轻触垫子“刷分”**：若 soft_landing_bonus 阈值过宽，agent 可能短暂满足条件后再次起飞，反复获取 bonus。需要监测单条轨迹中 bonus 是否多次触发。  
- **过度保守不敢行动**：若 stability_cost 权重过大，agent 可能原地不动或不敢使用引擎，导致始终无法靠近目标。