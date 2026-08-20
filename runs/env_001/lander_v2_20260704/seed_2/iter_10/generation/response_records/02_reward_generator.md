# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Potential-based shaping reward: reward = V(next_obs) - V(obs)
    where V(s) penalizes distance, speed, angle and angular velocity.
    This encourages the agent to move towards smaller distances and lower velocities.
    """
    def state_value(o):
        # Position penalty (L1 distance to target)
        pos_penalty = abs(o[0]) + abs(o[1])
        # Speed penalty (sum of L1 velocities)
        vel_penalty = abs(o[2]) + abs(o[3])
        # Orientation penalty
        ang_penalty = abs(o[4])
        # Angular velocity penalty
        angvel_penalty = abs(o[5])

        # Weighted combination; the scaling factors are chosen so that typical ranges
        # of the observables produce comparable contributions.
        return -pos_penalty - 0.5 * vel_penalty - 0.2 * ang_penalty - 0.1 * angvel_penalty

    v_obs = state_value(obs)
    v_next = state_value(next_obs)
    shaping = v_next - v_obs

    total_reward = shaping
    components = {"shaping_reward": shaping}
    return float(total_reward), components
```

# reward_v1 设计说明

## 奖励组件

仅使用一个组件：**shaping_reward**（基于势能差分的塑形奖励）。

- **数学形态**：`shaping_reward = V(next_obs) - V(obs)`，其中 `V(s) = −(|x|+|y|) − 0.5(|vx|+|vy|) − 0.2|angle| − 0.1|ang_vel|`。
- **角色**：主学习信号（密集引导）。该组件同时融合了距离接近、速度抑制、姿态稳定三个维度的信息，通过差分形式奖励每一步的状态改善，避免了单纯使用位置距离负奖励时常见的“冲撞后快速终结”局部最优问题。由于其差分语义，当 agent 处于已经很接近目标且低速度、姿态良好的状态时，不会持续累积大额奖励，从而降低了利用风险。
- **设计动机**：本轮重启要求提出 materially different 的设计假设。此前尝试过的 `distance_reward + landing_quality + stability_penalty` 以及其他骨架组合表现出高度不稳定性（得分从 -33 到 -112 剧烈波动），表明固定状态的持续惩罚/奖励结构容易陷入局部最优（例如过度悬停赚取 landing quality，或快速撞向目标以提前终止 episode）。新的势能差分设计从数学机制上改变了奖励的时间语义——它将关注点从“在某个状态获得多少分”转变为“这一动作让状态变好了多少”。这种差分形式天然具备更稠密且与动作关联更强的梯度，同时抑制了利用状态累积奖励的捷径。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

环境卡片明确声明 `explicit_success_flag_available = false` 和 `explicit_failure_flag_available = false`，并且 `info` 为空字典，没有任何可用的终止原因字段。因此无法可靠地识别成功或失败终止，不能设计基于 terminal flag 的奖励/惩罚。所有学习信号必须从观测中推导。

## 留到后续迭代的组件

- **energy_penalty / time_penalty**：先让 agent 学会稳定着陆，再考虑动作效率或时间优化。
- **soft_landing_proxy**：由于本版已通过势能差分自然地要求低速着陆，显式的接触条件组合奖励可暂缓加入，避免在多条件触发时引入不连续性。若训练后出现“接近但不接触”的徘徊现象，可再引入轻量的软着陆近似信号。
- **terminal_success_reward**：待环境 wrapper 明确暴露成功标志后加入，以强化最终着陆瞬间的奖励。
- **gated_reward**：目前不需要复杂门控。

## 训练后应观察的 failure mode

1. **目标附近震荡**：agent 可能在低距离/低速度状态下频繁前后微调以赚取正差分。需观察距离-速度曲线是否存在微小振荡，若出现则可能需要加入极小的动作代价或在势能函数中加入微量死区。
2. **低速悬停**：agent 可能学会在高处就保持低速缓慢下降，导致 episode 过长（仍在探索早期可能无此问题，但后续需要观察）。若发生，可后期加入 time_penalty 或使用速度上限条件惩罚。
3. **大角度撞击着陆**：势能函数中角度和角速度的权重相对较小，若 agent 学会用大角度高速撞击触地仍能获得一定的差分改善（因为距离迅速减小），则可能需要调高姿态项系数。
4. **侧向飘移**：`|x|` 和 `|vx|` 同时被惩罚，但若 agent 学会通过侧向移动来探索并偶尔获得差分奖励，可能导致水平方向控不准。需观察最终着陆点的水平偏差是否满足任务要求。
