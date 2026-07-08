# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 核心进步信号：鼓励沿前进方向快速移动
    forward_velocity = next_obs[2]
    w_fwd = 1.0
    forward_reward = w_fwd * forward_velocity

    # 姿态稳定约束：惩罚躯干倾角偏离直立
    hull_angle = next_obs[0]
    w_angle = 1.0  # 平方惩罚，对较大倾角敏感
    angle_penalty = -w_angle * (hull_angle ** 2)

    # 垂直稳定约束：抑制跳跃或剧烈起伏
    vertical_velocity = next_obs[3]
    w_vert = 0.5
    vert_penalty = -w_vert * (vertical_velocity ** 2)

    total_reward = forward_reward + angle_penalty + vert_penalty

    components = {
        'forward_reward': forward_reward,
        'angle_penalty': angle_penalty,
        'vert_penalty': vert_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 奖励组件与角色

| 组件 | 角色 | 数学形式 | 权重 |
|------|------|----------|------|
| `forward_reward` | **主学习信号**：直接奖励前进速度，驱动机器人尽快向前。每步都有连续梯度，稠密且任务核心。 | `1.0 * next_obs[2]` | 1.0 |
| `angle_penalty` | **姿态稳定约束**：惩罚躯干偏离直立，抑制倾倒倾向，防止单纯追求速度而失控。 | `-1.0 * (next_obs[0])²` | 1.0 |
| `vert_penalty` | **垂直运动约束**：惩罚过大的垂直速度，减少跳跃和剧烈起伏，促使步态平稳。 | `-0.5 * (next_obs[3])²` | 0.5 |

三个组件构成简单但完整的行走学习信号，符合 role‑based budget（1 主信号 + 2 稳定约束）。

## 为何未使用 terminal 奖励

环境未提供显式成功/失败标志（`info` 为空，且 `explicit_success_flag_available=false`、`explicit_failure_flag_available=false`）。我们无法在奖励函数中安全区分成功抵达终点或身体倾倒，强制加入 `terminal_success_reward` 或 `terminal_failure_penalty` 会引入不可靠的信号，违反设计原则。没有 `done` 标志，也未加入 `alive_bonus`，避免因缺少终止信息而产生错误存活奖励。

## 预留后续迭代的组件

- **能量/动作代价（energy_penalty）**：v1 阶段应优先学会稳定行走，不宜过早约束控制力矩。完成稳定前进后可在 v2 加入 `action square sum` 惩罚。
- **Gated 或 proxy 奖励**：目前无 success flag，若后续需要鼓励“接近终点”的行为，可利用 LIDAR 或速度阈值设计 soft proxy，但须在证明当前设计对远处地形引导不足时再引入。
- **动作平滑惩罚**：当前不含前一步动作，无法直接实现；若环境提供动作历史，后续可加入。

## 预期 failure mode 与观察重点

- **fast_then_fail**：若 agent 以牺牲稳定性为代价追求高速，倾角惩罚会逐渐收紧，但权重选择是关键。训练中应监控 `angle_penalty` 的量级，必要时增大 `w_angle` 或引入角速度惩罚。
- **hover_or_stand_still**：单纯的前向速度奖励不会诱使原地不动（因为没有存活奖励），但若倾角惩罚过强，可能导致 agent 不敢迈步。应观察首步执行情况，适时微调权重。
- **过度弹跳**：垂直速度惩罚抑制弹跳，但若地形不平坦，可能压制必要的迈步起伏，需通过试验观察步态是否僵硬。

整体上，这份 reward 设计优先保证学习信号稠密且可解释，为后续精调留出清晰空间。
