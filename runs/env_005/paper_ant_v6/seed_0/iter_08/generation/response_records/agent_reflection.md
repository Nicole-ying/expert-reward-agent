# Response Record

```markdown
# 设计理由
- 删除 `action_penalty`（每步 −0.285，约为前进奖励的 52%）和僵尸 `height_penalty`（active_rate 0%）。
- 新增 `height_health` gate：把高度上下界都做成软约束，乘到前进奖励上，填补环境终止条件 0.2 m 和 1.0 m 的前兆空缺。
- gate 形式：`lower = max(0, min((body_z - 0.2)/0.1, 1.0))`，`upper = max(0, min((1.0 - body_z)/0.1, 1.0))`，健康区间 [0.3, 0.9] 内 gate=1；低于 0.3 或高于 0.9 时线性衰减，在终止边界（0.2/1.0）恰好为 0。
- 系数校准：gate 不引入新惩罚，不会主导 reward；前进奖励 per-step ≈0.55，gate 在健康区间为 1，影响轻微；gate 在最不理想但仍安全的位置（如 0.25 m）仍 ≥0.5，不会切断梯度。
- 同时保留 `lateral_penalty` 和 `descend_penalty`，它们提供侧向和下降速度的温和约束，不干扰主体修改。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    vx = next_obs[13]
    vy = next_obs[14]
    vz = next_obs[15]

    # 姿态健康因子（连续，保持直立）
    pose_error = quat_x**2 + quat_y**2 + quat_z**2
    posture_factor = 2.718281828 ** (-10.0 * pose_error)

    # 高度健康 gate（上、下界前兆）
    lower_safe = max(0.0, min((body_z - 0.2) / 0.1, 1.0))   # 0.2→0, 0.3→1
    upper_safe = max(0.0, min((1.0 - body_z) / 0.1, 1.0))   # 1.0→0, 0.9→1
    height_factor = lower_safe * upper_safe

    # 前进奖励，同时受姿态和高度 gate 调制
    base_forward = max(0.0, vx)
    forward_reward = 1.0 * base_forward * posture_factor * height_factor

    # 侧向速度惩罚
    lateral_penalty = -0.5 * (vy ** 2)

    # 下降速度惩罚：阻止过快坠落
    descend_penalty = -0.2 * max(0.0, -0.5 - vz)

    total_reward = forward_reward + lateral_penalty + descend_penalty

    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "descend_penalty": descend_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 高度上限前兆缺失，height_penalty 僵尸（0% active），action_penalty 可能是 len 断崖暴跌的根因。
- **evidence**: len=17.05，terminated 100%；height_penalty active=0%；action_penalty per-step −0.285 vs forward 0.55。
- **behavior**: agent 快速失败，无法存活。
- **signal**: 高度上限软约束缺位；动作惩罚过强抑制生存探索。
- **level**: Level 2 — 结构变换，添加 height_health gate，移除 action_penalty 和僵尸 height_penalty。
- **hypothesis**: 去掉动作惩罚可恢复探索能力，高度 gate 提供上下界梯度，episode 长度应显著回升并带动 forward 得分。
- **risk**: 无动作惩罚可能导致能量效率低或步态抖动；若噪声增大，后续可添加关节速度平滑项。
```
