# Response Record

## 分析与修改理由

**当前智能体行为：** 训练得到的固定策略几乎不移动——每回合进度均值为 0.754，接触奖励永远为 0，角度/速度惩罚几乎不触发（active_rate ≈ 0）。episode 长度中位数 989 步，19/20 回合为截断（超时），仅 1 次真正终止（很可能也是因为随机飘移出界）。这说明智能体学会了“空中悬停”来避免失败，但完全没有向目标降落的意图。

**核心问题诊断：**
- 进度奖励 `w_progress * (prev_dist - next_dist)` 的系数 1.0 过小，在初始阶段距离变化微弱时提供的梯度不足，无法引导下降行为。
- 接触奖励的质量因子虽然合理，但在智能体从未接触的情况下完全无效，无法成为早期探索的驱动力。
- 速度/角度惩罚 active_rate 极低、幅度极小，它们既没有阻碍漂浮，也没有产生有用的行为约束。

**最佳历史（iter 4, score 140.27）的关键特征：** 该版本有显著的 contact_reward（mean ≈ 7.2）和中等 progress（mean ≈ 0.065），说明其奖励结构曾经成功引导过接触着陆。我们应以该结构为基础，但必须增强“非接触阶段”的下降驱动力，以避免早期训练陷入悬浮的局部最优。

**修改方案（以 best 结构为底，做出有证据的修改）：**
1. **大幅提高 progress 系数**：从 1.0 提升至 **10.0**，使每一步的距离缩短产生更强正反馈。
2. **引入持续性距离惩罚**：加入 `-0.1 * next_dist`，为“远离目标”的状态提供持续负压，防止长时间悬浮。
3. **保留接触奖励**：保持原有权重（50）和 quality 的计算方式（hinge 形式），确保着陆质量。
4. **加入极轻的燃料惩罚**：对点燃主引擎的动作（action==2）施加 `-0.2`，鼓励执行器效率，同时不阻断必要推力。
5. **移除原有速度/角度惩罚**：它们未激活，且可能干扰下降阶段的动力学探索。

这些改动旨在**用更密集的下降信号覆盖整个 episode**，使智能体必须在接近目标与节省燃料之间做出权衡，从而打破悬浮平衡。

## 新奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- weights & thresholds -----
    w_progress = 10.0
    w_contact = 50.0
    v_target = 0.3             # desired landing speed
    a_target = 0.2             # desired landing angle (rad)
    w_distance_penalty = 0.1
    w_fuel_penalty = 0.2

    # ----- distance progress (always active) -----
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = w_progress * (prev_dist - next_dist)

    # ----- continuous distance penalty (drives downward motion) -----
    distance_penalty = -w_distance_penalty * next_dist

    # ----- contact quality (only when both legs touch) -----
    contact = next_obs[6] * next_obs[7]
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])

    q_speed = max(0.0, 1.0 - speed / v_target)
    q_angle = max(0.0, 1.0 - angle / a_target)
    quality = q_speed * q_angle
    contact_reward = w_contact * contact * quality

    # ----- fuel penalty (only for main engine) -----
    fuel_penalty = -w_fuel_penalty if action == 2 else 0.0

    # ----- total reward -----
    total_reward = progress + distance_penalty + contact_reward + fuel_penalty

    components = {
        'progress': progress,
        'distance_penalty': distance_penalty,
        'contact_reward': contact_reward,
        'fuel_penalty': fuel_penalty
    }
    return float(total_reward), components
```
