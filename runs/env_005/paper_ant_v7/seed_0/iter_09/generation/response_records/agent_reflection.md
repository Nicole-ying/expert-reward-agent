# Response Record

# 设计理由
移除 `upright_penalty`（active_rate 0.7%，episode_sum_mean -0.27，几乎不活跃的僵尸组件），替换为 `lateral_velocity_penalty`，对横向速度进行连续惩罚。这引导四足机器人减少左右摇摆，将更多能量集中于前进方向，从而在不牺牲生存能力的前提下提升有效前进速度。系数选择 -0.2，使得 per-step 惩罚预估约 -0.04，远小于主信号 per-step 9.2 的 0.5 倍，不会压垮探索。此改动保留原有的前向凸化奖励、高度门控、边界 penalty 及力矩成本，仅增强方向效率。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    torque_penalty = sum([a**2 for a in action])

    # ---------- 身体直立投影 ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 高度安全门控 ----------
    low_factor = max(0.0, (body_z - 0.2) / 0.1)
    high_factor = max(0.0, (1.0 - body_z) / 0.1)
    height_factor = min(low_factor, high_factor)
    height_reward = 0.1 * height_factor

    # ---------- 高度边界前兆惩罚（保留，虽然已接近失效但仍作为安全网） ----------
    low_penalty = -0.3 * max(0.0, (0.25 - body_z) / 0.05)
    high_penalty = -0.3 * max(0.0, (body_z - 0.95) / 0.05)
    height_boundary_penalty = low_penalty + high_penalty

    # ---------- 主学习信号：凸化前进速度 ----------
    forward_reward = 1.0 * body_x_velocity + 0.5 * max(0.0, body_x_velocity)**2

    # ---------- 横向速度惩罚（新组件，替换僵尸 upright_penalty） ----------
    lateral_velocity_penalty = -0.2 * abs(body_y_velocity)

    # ---------- 力矩效率约束 ----------
    action_cost = -0.05 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = (forward_reward + height_reward + height_boundary_penalty
                    + lateral_velocity_penalty + action_cost)

    components = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "height_boundary_penalty": height_boundary_penalty,
        "lateral_velocity_penalty": lateral_velocity_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前奖励函数缺少横向运动约束，两个惩罚组件（height_boundary_penalty, upright_penalty）活跃率极低（<1%），实为僵尸组件。
- **behavior**: agent 已掌握高速、安全行走（702 步均长，50% 终止），但可能因左右摆动损失前进效率。
- **signal**: 缺少抑制横向速度的惩罚，能量无法充分转化为前进速度。
- **level**: Level 2（结构变换：删除僵尸 upright_penalty，新增连续横向惩罚）
- **hypothesis**: 对 body_y_velocity 施加轻度惩罚后，agent 将调整步态减少侧向摆动，用相同力矩获得更高 body_x_velocity，推动总得分向 2000 靠近。
- **risk**: 若系数过大可能过度限制必要的横向稳定调节，导致摔倒增加；当前系数 -0.2 极轻，风险低。移除 upright_penalty 可能让极少数已倒地 episode 更早结束，但历史数据中该组件影响可忽略。
