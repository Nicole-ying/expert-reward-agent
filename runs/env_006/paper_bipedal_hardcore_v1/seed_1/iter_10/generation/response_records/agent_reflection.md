# Response Record

# 设计理由
本轮为正常模式，仅修改一个组件。按第 0 步审计：
- 终止模式：20 个 episode 中 19 个 terminated（推测为摔倒），仅 1 个 truncated；len=370 说明 agent 能存活一段距离但最终失败。
- 观测使用扫描：已使用 obs[0]（倾角）、obs[2]（水平速度）、obs[12]/obs[13]（脚触地）。未使用的重要信号包括动作幅值（可表征能耗）和 hull_angular_velocity 等。历史最佳 iter1（score=-18.01, len=411）含有 `action_efficiency_penalty`，之后该信号被移除，分数再未恢复。
- 僵尸组件：`ground_penalty` 的 active_rate 仅 0.1%，几乎不触发，说明双脚同时离地极少发生，该组件对训练无实质贡献，可安全替换。
- 信号缺口：缺少动作效率约束，agent 可能以高力矩、高能耗方式行走，不仅降低真实评分，也会因剧烈动作引发摔倒。

因此采用 Level 2 变换：删除失效的 `ground_penalty`，新增 `action_penalty`（动作幅值惩罚），使用动作本身作为 proxy。系数校准：期望步均惩罚约 0.02，远小于主前进信号步均 0.31，符合 ≤0.3× 的负担上限。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测信号
    hull_angle = obs[0]
    horizontal_speed = obs[2]

    # 组件 A：前进主奖励
    forward_progress = horizontal_speed  # 系数 1.0

    # 组件 B：身体倾角稳定性惩罚
    angle_threshold = 0.5
    w_angle = 5.0
    angle_error = max(0.0, abs(hull_angle) - angle_threshold)
    stability_angle_penalty = -w_angle * (angle_error ** 2)

    # 组件 C：动作效率惩罚（替代失效的 ground_penalty）
    w_action = 0.02
    action_sum_sq = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    action_penalty = -w_action * action_sum_sq

    total_reward = forward_progress + stability_angle_penalty + action_penalty

    components = {
        "forward_progress": forward_progress,
        "stability_angle_penalty": stability_angle_penalty,
        "action_penalty": action_penalty,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺口明显——缺少动作效率约束，且 ground_penalty（active_rate 0.1%）为僵尸组件，可安全替换。
- **behavior**: 存活长度中等（~370），最终因摔倒终止，前进速度尚可但可能以高能耗步态浪费力矩。
- **signal**: 缺少动作幅值惩罚；历史最佳 iter1 存在该信号，移除后分数长期未能恢复。
- **level**: Level 2
- **hypothesis**: 加入小系数动作幅值惩罚，引导策略选择低力矩、高效步态，降低剧烈动作引发的摔倒概率，从而提升真实得分与存活长度。
- **risk**: 系数极小，若 agent 当前依赖大幅摆动维持平衡，可能短期略微降低前进速度；但历史证据显示动作惩罚曾与前进速度共存，风险可控。
