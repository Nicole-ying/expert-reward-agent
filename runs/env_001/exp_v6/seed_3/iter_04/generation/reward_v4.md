## 诊断

### 1. Agent 发生了什么？
- **Score=-101.92, len=86.8, 100% early termination** — agent 在 crash，活不到 150 步。
- 步均总奖励仅 0.007，信号极其微弱。agent 被迫在近乎零信号的环境中学习。

### 2. 哪个组件是主因？
- `progress_delta` mean=0.014 — **进步信号本身太小**，每步只移动 ~0.014 单位距离。
- `stability_penalty` ratio_to_progress=-0.50 — 惩罚吃掉了一半的进步信号，agent 收到矛盾指令：「靠近目标」vs「别动」。
- 但历史最佳 (iter 2) 的 penalty ratio 仅 -0.07、score 仍然 -96。说明**即使 penalty 很轻，纯 progress_delta 也不足以教会 agent 软着陆**——它只知道往目标冲，不知道要减速，最终 crash。

### 3. 我之前改了什么？
- Iter 2→3：从 iter 2 的轻 penalty (0.001) 改为距离门控 + 10x 基础权重 (0.01)，希望「远处自由，近处精细」。结果 penalty 均值从 -0.001 反弹到 -0.007，ratio 从 -0.07 恶化到 -0.50，得分倒退。
- **问题本质**：progress_delta 只有一个维度（靠近），agent 学不到「减速」和「姿态控制」。纯 penalty（惩罚绝对值）提供的是负反馈——告诉 agent「别这样」而非「往这个方向改进」。

### 修改方案：从 penalty 切换到 potential-based shaping 的多项势能

改用 `potential_based_shaping` 骨架的推荐形态：**Φ = -(distance + α·speed + β·|angle| + γ·|angvel|)**。reward = Φ(next) - Φ(obs)。

展开后：`progress_delta + α·(speed_obs - speed_next) + β·(|angle_obs| - |angle_next|) + γ·(|angvel_obs| - |angvel_next|)`

这不再是「惩罚当前状态」，而是**奖励所有维度的改进**——减速有奖、调姿有奖、靠近有奖。所有信号同向，不存在 ratio 对抗问题。这是层次 2（改数学形式），仍在 potential-based 骨架家族内。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- potential-based shaping: Φ = -(distance + w_speed*speed + w_angle*|angle| + w_angvel*|angvel|) ----
    # reward = Φ(next_obs) - Φ(obs) = sum of improvements across all dimensions.
    # This replaces the penalty paradigm: instead of punishing bad states,
    # we reward movement toward better states in every dimension simultaneously.

    # --- positions ---
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5

    # --- speeds (L1 norm for simplicity) ---
    speed_obs = abs(obs[2]) + abs(obs[3])
    speed_next = abs(next_obs[2]) + abs(next_obs[3])

    # --- angle ---
    angle_obs = abs(obs[4])
    angle_next = abs(next_obs[4])

    # --- angular velocity ---
    angvel_obs = abs(obs[5])
    angvel_next = abs(next_obs[5])

    # --- potential values ---
    w_speed = 0.05
    w_angle = 0.05
    w_angvel = 0.02

    phi_obs = -(dist_obs + w_speed * speed_obs + w_angle * angle_obs + w_angvel * angvel_obs)
    phi_next = -(dist_next + w_speed * speed_next + w_angle * angle_next + w_angvel * angvel_next)

    # --- reward: improvement in potential ---
    total_reward = phi_next - phi_obs

    # --- decompose for logging ---
    progress_delta = dist_obs - dist_next  # distance improvement
    speed_reduction = w_speed * (speed_obs - speed_next)  # speed improvement
    angle_reduction = w_angle * (angle_obs - angle_next)  # angle improvement
    angvel_reduction = w_angvel * (angvel_obs - angvel_next)  # angvel improvement

    components = {
        'progress_delta': progress_delta,
        'speed_reduction': speed_reduction,
        'angle_reduction': angle_reduction,
        'angvel_reduction': angvel_reduction,
        'total_reward': total_reward,
    }

    return float(total_reward), components
```