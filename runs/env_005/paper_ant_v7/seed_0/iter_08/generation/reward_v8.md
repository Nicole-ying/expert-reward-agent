# 设计理由

第7轮将 `upright_penalty` 从 hinge（仅在大倾斜时激活）改为全时的二次惩罚 `-0.5 * (1 - body_up_z)²`，结果 score 从 662.91 崩塌到 27.15，len 从 621.95 降到 404.2，terminated 率升高。原因在于二次惩罚对任意姿态偏差都施加压力，迫使 agent 为维持极端直立而牺牲前进速度，甚至抑制了步态探索。尽管 per‑step 惩罚看似不大，但它无处不在，扼杀了第6轮 hinge 下允许的自然摇摆与速度提升。

本轮将 `upright_penalty` 改回 hinge 形式，但采用比之前更精确的门控：  
- 身体直立投影 `body_up_z` 在安全区（≥ 0.8）时完全不罚，允许自然步态与躯干轻微摆动。  
- 当 `body_up_z` 跌破 0.8 后，按线性增长惩罚，促使策略迅速恢复直立，避免倾斜积累导致跌倒（最终高度出界）。  
- 系数 0.5 确保即使身体躺平（`body_up_z=0`）惩罚也仅 –0.4，不会超过前进奖励的 20 %；正常步态下此惩罚几乎为 0，回归第6轮“前进奖励与适度姿态约束”的平衡。

# 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    torque_penalty = sum([a**2 for a in action])

    # ---------- 身体直立投影 ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 高度安全门控 ----------
    low_factor = max(0.0, (body_z - 0.2) / 0.1)
    high_factor = max(0.0, (1.0 - body_z) / 0.1)
    height_factor = min(low_factor, high_factor)
    height_reward = 0.1 * height_factor

    # ---------- 高度边界前兆惩罚 ----------
    low_penalty = -0.3 * max(0.0, (0.25 - body_z) / 0.05)
    high_penalty = -0.3 * max(0.0, (body_z - 0.95) / 0.05)
    height_boundary_penalty = low_penalty + high_penalty

    # ---------- 主学习信号：凸化前进速度 ----------
    forward_reward = 1.0 * body_x_velocity + 0.5 * max(0.0, body_x_velocity)**2

    # ---------- 直立姿态 hinge 惩罚 (代替全时二次惩罚) ----------
    # 仅当身体向上投影低于 0.8 时激活，线性惩罚促使其恢复直立
    upright_hinge = max(0.0, 0.8 - body_up_z)
    upright_penalty = -0.5 * upright_hinge

    # ---------- 力矩效率约束 ----------
    action_cost = -0.05 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = (forward_reward + height_reward + height_boundary_penalty
                    + upright_penalty + action_cost)

    components = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "height_boundary_penalty": height_boundary_penalty,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: `height_boundary_penalty` 僵尸（active_rate 0.7%），终止前兆缺失；`upright_penalty` 全时二次惩罚过强，压制了前进速度。
- **behavior**: agent 为减少姿态惩罚而牺牲前进速度，倾向站立 / 慢速移动，导致生存时间缩短，terminated 率高。
- **signal**: 需恢复对轻微倾斜的宽容度（hinge 门控），让自然步态能够产生足够的前进速度。
- **level**: Level 2
- **hypothesis**: 去掉无处不在的姿态压力后，agent 将重新获得速度提升空间，前进奖励会驱动生存与高效步态，使总得分恢复到第6轮水平并进一步增长。
- **risk**: 若 hinge 阈值 0.8 仍过严，倾斜累积可能导致高度终止率不降反升；必要时可微调阈值至 0.75。