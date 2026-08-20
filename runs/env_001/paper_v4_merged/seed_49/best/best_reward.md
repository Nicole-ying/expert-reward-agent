1. `evidence`：progress 是唯一活跃组件，angle_penalty 和 angvel_penalty 触发率为 0，一半 ep 提前终止，score=-53.8，表明缺少速度约束导致撞击失败。
2. `behavior_diagnosis`：Agent 已学会朝目标移动以获取 progress 奖励，但未减速，因此以破坏性速度着陆而 crash。
3. `signal_completeness`：缺少对速度过快的反馈，成功/失败终端信号不可达（无 done 信息），但可从 obs[2]、obs[3] 构造连续安全约束。
4. `selected_level`：Level 2，证据是僵尸组件占位、未用观测维度 vx/vy 存在、active_rate 0%，触发结构替换。
5. `selected_intervention`：移除 angle_penalty 与 angvel_penalty，新增 overspeed_penalty，对 √(vx²+vy²) 超过 0.5 的部分以 -0.005 线性惩罚。
6. `falsifiable_hypothesis`：加入速度惩罚后，Agent 被迫降低速度来避免惩罚，从而减少高速碰撞，episode 长度应增加，early_terminal 数量下降。
7. `expected_next_round`：overspeed_penalty 的 active_rate 将大于 0，episode 平均长度上升，early_terminal 比例下降，score 可能小幅改善。
8. `main_risk`：惩罚抑制移动意愿，导致 Agent 停留在远处安全区，progress 下降，episode 虽长但无进展。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]

    # ---------- 1) Main progress: distance reduction ----------
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    delta_dist = dist - next_dist
    progress = 1.0 * delta_dist

    # ---------- 2) Overspeed penalty (replaces inactive angle/angvel) ----------
    speed = (vx ** 2 + vy ** 2) ** 0.5
    safe_speed = 0.5
    overspeed = max(0.0, speed - safe_speed)
    overspeed_penalty = -0.005 * overspeed

    # ---------- Total reward ----------
    total_reward = progress + overspeed_penalty

    components = {
        'progress': progress,
        'overspeed_penalty': overspeed_penalty
    }
    return float(total_reward), components
```