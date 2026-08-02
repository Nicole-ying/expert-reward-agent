# 设计理由
本轮仅做 Level 2 结构变更：**新增垂直速度异常惩罚组件**。

**审计结论**：当前奖励函数已覆盖水平速度、身体倾角、角速度、动作代价、双脚离地惩罚，但缺少对坠落式垂直速度的约束。从训练反馈看，episode 普遍以 terminated（摔倒）结束，而摔倒是机器人无法应付地形时的终局。观测中垂直速度 `obs[3]` 完全未被使用，它可以直接反映机器人是否正在失控坠落——摔倒前的典型征兆是 hull_angle 快速增大伴随向下的垂直速度突变。现有 `posture_penalty` 仅在倾角 >0.3 rad 时触发（active 仅 4.3%），说明多数摔倒可能在倾角快速穿越阈值的过程中未能被有效惩罚/预警，而垂直速度是一个更直接、更前置的危险信号。

因此，引入 `vertical_speed_penalty`，采用有界 hinge 形式：仅惩罚过大的向下速度，阈值 `2.0`（根据常见双足机器人下坠幅度设定，正常跳跃落地一般在此以下），系数 `-0.08`，使得在触发时（例如垂直速度 -5 m/s，差值为 3）惩罚约 -0.24，但 active rate 预期极低（仅失控坠落时），per-step 负担远小于主信号 `progress_reward` 的 0.3x。此信号可在摔倒前数步提供预警，鼓励 agent 在下降过快时调整姿态或步频，减少最终摔倒。

**为什么只改这一个组件？**  
其他组件功能已基本合理：`air_penalty` 虽 active 71.2% 但历史上加入后 len 和 score 均改善，暂时保留；`posture_penalty` 阈值可能稍宽松但 active 极低说明姿态保持良好；`ang_vel_penalty` 贡献可忽略但无危害。本轮聚焦在填补明显的感知缺口，避免同时变更多个组件导致信号震荡。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：向前速度（线性）
    horizontal_speed = next_obs[2]
    progress = 2.0 * horizontal_speed

    # 稳定/安全约束：姿态角度超出健康范围时软惩罚（hinge）
    hull_angle = next_obs[0]
    max_allowed_angle = 0.3
    posture_penalty = -5.0 * max(0.0, abs(hull_angle) - max_allowed_angle)

    # 稳定/安全约束：角速度惩罚，抑制剧烈摇晃
    ang_vel = next_obs[1]
    ang_vel_penalty = -0.05 * (ang_vel ** 2)

    # 效率/动作代价：轻微二次惩罚
    action_cost = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # 空中惩罚：基于地面接触信号，抑制双脚同时离地
    contact_sum = next_obs[12] + next_obs[13]
    air_penalty = -0.2 * max(0.0, 1.0 - contact_sum)

    # 新增：垂直速度异常惩罚，仅惩罚过快的向下坠落（hinge）
    vertical_speed = next_obs[3]
    max_allowed_fall = 2.0  # 正常跳跃落地的下降速度极限
    vertical_speed_penalty = -0.08 * max(0.0, -vertical_speed - max_allowed_fall)

    total_reward = (progress +
                    posture_penalty +
                    ang_vel_penalty +
                    action_cost +
                    air_penalty +
                    vertical_speed_penalty)

    components = {
        'progress_reward': progress,
        'posture_penalty': posture_penalty,
        'ang_vel_penalty': ang_vel_penalty,
        'action_cost': action_cost,
        'air_penalty': air_penalty,
        'vertical_speed_penalty': vertical_speed_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 垂直速度 obs[3] 未被使用，而它是摔倒前的关键前置信号；补齐此缺口应能在摔倒前更早施加压力。
- **behavior**: agent 平均存活 323 步、水平速度约 0.27，但 19/20 episode 以摔倒终止；进步有但无法避免终局坠落。
- **signal**: 缺少对失控坠落速度的直接惩罚，导致 agent 仅在角度过大时受罚（触发过晚），错过预先调节的窗口。
- **level**: Level 2
- **hypothesis**: 对过大的向下垂直速度施加软惩罚，可引导 agent 在开始快速下落前调整步态/姿态，减少最终摔倒的概率，拉长 episode 并累积更多 progress_reward。
- **risk**: 阈值若偏高可能惩罚正常跳跃落地，导致 agent 不敢越过坑洼；但阈值 2.0 保守且系数低，预期 active rate 极低，副作用有限。