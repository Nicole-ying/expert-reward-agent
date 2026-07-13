# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 奖励权重（可调，当前基于典型环境尺度）
    progress_weight = 1.0
    stability_angle_weight = 1.0
    stability_angvel_weight = 0.1

    # 主学习信号：前进速度
    forward_velocity = next_obs[2]          # 水平方向速度 (m/s)
    progress_reward = progress_weight * forward_velocity

    # 稳定约束：身体倾斜与旋转速度
    hull_angle = next_obs[0]                # 身体与竖直方向夹角 (rad)
    hull_ang_vel = next_obs[1]              # 身体角速度 (rad/s)
    stability_penalty = -(stability_angle_weight * abs(hull_angle) +
                          stability_angvel_weight * abs(hull_ang_vel))

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 奖励组件与角色
本方案使用两个组件，符合 v1 的 role‑based component budget (1 个主学习信号 + 1 个稳定约束)。

- **progress_reward（主学习信号）**：直接奖励下一时刻的水平速度 `next_obs[2]`。它是每步都有梯度的稠密信号，鼓励智能体快速向前移动，形成行走驱动力。
- **stability_penalty（平稳定约束）**：惩罚身体倾斜角度 `next_obs[0]` 和角速度 `next_obs[1]` 的绝对值。抑制大幅度倾斜和快速旋转，引导智能体保持直立姿态，降低摔倒风险。

两个组件联合作用：智能体为了获得高的 progress_reward 必须产生足够的前向速度，同时为了避免 stability_penalty 又必须保持姿态稳定，从而促使生成“快而稳”的步态。

## 2. 为何没有使用 terminal_success_reward / terminal_failure_penalty
- 环境卡片声明 `explicit_success_flag_available: false` 和 `explicit_failure_flag_available: false`，且 `info` 字典为空，无法判断终止原因。
- `compute_reward` 的参数中不包含 `done` 标志，因此无法在奖励函数中实现基于终止条件的奖励或惩罚。
- 违反了信号可用性优先原则，故不引入。

## 3. 留待后续迭代的组件
- **energy_penalty**（动作幅度惩罚）：v1 阶段优先学会行走，过早加入可能让智能体“害怕”发力而静止不动。
- **action_smoothness_penalty**：因缺乏历史动作信息当前不可用，但后续若传入 `last_action` 可加入以抑制关节抖震。
- **接触模式奖励**（如交替着地、防止双脚离地）：步态精细化，v1 先不约束，避免限制探索。
- **效率类组件**（如时间惩罚或基于激光雷达的地形适应奖励）：需在稳定行走基础上引入，后续阶段考虑。

## 4. 预期 failure mode 与观察要点
训练后应重点观察以下失败模式：
- **fast_then_fail**：如果 stability_penalty 权重不足，智能体可能过度追求高速度，在加速后很快摔倒。表现为 episode 长度短、前进距离小但步均速度高。
- **hover_or_stand_still**：如果稳定性惩罚过重或 progress_reward 太小，智能体可能选择几乎不动（速度接近零、角度近乎零）来避免惩罚，导致 episode 很长但几乎没有位移。
- **over_conservative_policy**：智能体以极小步幅缓慢前进，速度始终很低，回报增长缓慢。

若观察到上述问题，可优先调整两个组件的权重比例，而非增加新组件。例如若出现 `fast_then_fail`，适当增大 `stability_angle_weight`；若出现 `hover_or_stand_still`，则减小稳定性惩罚或增大 `progress_weight`。
