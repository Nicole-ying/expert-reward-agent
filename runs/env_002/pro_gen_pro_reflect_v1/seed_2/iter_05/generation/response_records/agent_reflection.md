# Response Record

# 设计理由
当前奖励函数已经成功完成目标（平均分 320.21 > 300），agent 能够稳定走到地形终点。从组件表看，`stability_penalty` 每回合平均仅 -0.032，几乎不触发，说明姿态控制优秀；`energy_penalty` 每回合 -9.93，占比 1.9%，对速度有一定抑制作用但幅度很小。为了在不破坏现有稳定性的前提下进一步推动更快前进，我们采取 **Level 1 尺度修复**：将能量惩罚系数 `w_energy` 从 0.01 减半到 0.005。这样每步惩罚负担从 ~-0.012 降至 ~-0.006，给予关节力矩略微更大的自由度，可能让 agent 在保持姿态的前提下迈出更大步幅，从而提升前进速度。该修改不改变任何组件结构，只做系数微调，风险极低。

校准：主信号 per-step ≈ 0.603，惩罚 per-step ≈ 0.006，远小于 0.3×主信号，满足设计约束。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ====== 主学习信号：线性前进速度奖励 ======
    forward_vel = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_vel

    # ====== 姿态稳定性惩罚（hinge 门控） ======
    hull_angle = obs[0]
    hull_angvel = obs[1]
    vertical_vel = obs[3]

    # 倾角安全阈值：设为 0.4 rad（约 23°），在该范围内不施加惩罚
    safe_angle = 0.4
    angle_violation = max(0.0, abs(hull_angle) - safe_angle)

    w_angle = 1.5
    w_angvel = 0.01
    w_vertical = 0.01

    stability_penalty = -(
        w_angle * angle_violation ** 2 +
        w_angvel * hull_angvel ** 2 +
        w_vertical * vertical_vel ** 2
    )

    # ====== 能量效率惩罚（系数减半以给予更积极的运动自由度） ======
    w_energy = 0.005  # 原 0.01
    energy_penalty = -w_energy * (
        action[0] ** 2 + action[1] ** 2 +
        action[2] ** 2 + action[3] ** 2
    )

    total_reward = forward_reward + stability_penalty + energy_penalty
    components = {
        'forward_reward': forward_reward,
        'stability_penalty': stability_penalty,
        'energy_penalty': energy_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 终止条件摔倒已有前兆倾角惩罚但幅度极小，任务目标前进速度有直接梯度，无僵尸组件，整体缺口很小。
- **behavior**: agent 已能稳定高速走到终点，全部 episode 正常终止（成功），速度主导奖励。
- **signal**: 能量惩罚略抑制最高速度，可减弱以鼓励更激进的步态。
- **level**: Level 1
- **hypothesis**: 减半能量惩罚系数后，关节动作更自由，agent 能在保持姿态的前提下提高步频或步幅，从而进一步提升前进速度与总分。
- **risk**: 可能略微增大关节力矩，但在现有强稳定性惩罚的保护下，摔倒风险极低。若速度无提升，分数维持在现有水平。
