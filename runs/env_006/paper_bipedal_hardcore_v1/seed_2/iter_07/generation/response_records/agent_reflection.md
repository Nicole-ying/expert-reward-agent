# Response Record

# 设计理由

迭代 6 移除了铰链姿态惩罚（hinge_penalty），仅依赖双因子门控（角度 + 角速度）调制前进奖励来间接抑制摔倒。训练结果（score = -59.5，len = 380）明显弱于迭代 5（带 hinge_penalty 的 -52.46），且 18/20 的 episode 仍以摔倒终止。信号覆盖审计显示：hull_angle 已被用于门控，但缺少对接近摔倒状态的直接负向激励；leg ground contact 和 vertical_speed 等潜在有用维度仍未被使用，但历史实验（iter 2–4）中对接触惩罚的处理不当导致长度崩溃，暂不适合作为本次唯一切入点。

因此，本轮修改选择 **Level 2 结构变换 — 添加一个组件**：重新引入基于 hull_angle 的 hinge_penalty。该组件在 |hull_angle| 超过安全阈值 0.3 rad 后施加线性惩罚，直接告诫 agent 保持低倾斜角。配合已有的双因子门控（角度 + 角速度继续压低前进奖励），形成“前进减速 + 直接姿态惩罚”的双重防护，更清晰地表达“避免摔倒”的目标。

数学形式：  
`hinge_penalty = -0.5 * max(0, abs(hull_angle) - 0.3)`

系数校准：
- 主信号 per-step ≈ forward_reward / len ≈ 61.4 / 380 ≈ 0.16。
- 新惩罚在常见偏航角 0.4 rad 时的 per-step 惩罚约 0.05，占主信号的 31%，略超 0.3× 指导线但仍在可接受范围，且仅在较大倾角时触发，平均 per-step 期望约 0.02，与 energy_penalty（-0.015）合计约 0.035，远低于主信号的 0.5×（0.08），不会压垮学习。
- 阈值 0.3 rad 设在假设终止阈值 0.5 rad 的 60%，符合设计校准要求。

预期效果：agent 获得清晰的“保持小倾角”的梯度信号，减少致命摔倒的发生概率，推动 episode 长度和得分回升至迭代 5 以上水平。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 基本观测提取 ----------
    hull_angle = obs[0]                 # 身体倾角 (rad)
    hull_angular_velocity = obs[1]      # 身体角速度 (rad/s)
    horizontal_speed = obs[2]           # 质心水平速度

    # 只奖励正向移动
    forward_vel = max(0.0, horizontal_speed)

    # ---------- 1. 主学习信号：动态 gate-controlled forward progress ----------
    # gate 由角度和角速度共同决定，剧烈摇晃 → gate 变小 → 前进奖励骤减
    beta_angle = 2.0
    beta_angvel = 1.0
    gate = 1.0 / (1.0 + beta_angle * abs(hull_angle) + beta_angvel * abs(hull_angular_velocity))
    forward_reward = 1.0 * forward_vel * gate

    # ---------- 2. 效率约束：轻量力矩代价 ----------
    action_cost = sum(action[i] ** 2 for i in range(4))
    energy_penalty = 0.01 * action_cost

    # ---------- 3. 姿态铰链惩罚：接近摔倒的直接信号 ----------
    # 阈值 0.3 rad（约为终止条件 0.5 rad 的 60%），超出后线性惩罚
    hinge_penalty = 0.5 * max(0.0, abs(hull_angle) - 0.3)

    # ---------- 汇总 ----------
    total_reward = forward_reward - energy_penalty - hinge_penalty

    components = {
        "forward_reward": forward_reward,
        "energy_penalty": -energy_penalty,
        "hinge_penalty": -hinge_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号可用但缺少对摔倒的直接惩罚；ground contact/vertical speed 仍闲置但历史介入粗暴，本轮优先回补已证明有效的 hinge 信号。
- **behavior**: agent 存活较长但普遍以摔倒终止，依靠门控减速不足以彻底避免致命倾角。
- **signal**: 缺失对 hull_angle 的显式铰链惩罚，导致在 gate 削弱奖励后 agent 仍可能放任倾角增大。
- **level**: Level 2
- **hypothesis**: 加回 hinge_penalty 提供明确的“保持小倾角”梯度，与双因子门控配合，能进一步降低摔倒率，提升平均得分至迭代 5 以上。
- **risk**: 阈值附近刚触发惩罚时，combined penalty 可能轻微拖慢进步速度；若 agent 错误地抑制步幅以换取安全，可通过后续微调系数或调整 gate 敏感度缓解。
