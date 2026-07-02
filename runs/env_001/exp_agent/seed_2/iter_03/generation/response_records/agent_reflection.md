# Response Record

## 诊断

### 1. Agent 发生了什么？

**Hovering exploit（悬停利用）**。episode 长度 694.5（接近上限），soft_landing_proxy 的 nonzero_rate 高达 51.5%，这意味着 agent 大约一半的步数都在满足"近目标 + 低速 + 直立 + 双腿着地"的二值条件，每步收割 0.5 奖励。它学会了在目标附近保持一个"看起来像着陆"的姿态，而不是真正完成干净着陆。

### 2. 哪个组件是主要原因？

**soft_landing_proxy**：ratio_to_progress = 87.6x，mean=0.257 vs progress mean=0.003。进度信号在总奖励中几乎不可见——整个学习完全被二值 proxy bonus 主导。问题本质：二值条件 `if (A and B and C and D): bonus=0.5` 在阈值边界处无梯度，一旦 agent 学会了刚好满足四个条件，就不再有任何动力改进。

### 3. 上一轮改了什么？

上一轮（iter 2）把 stability_penalty 系数从 0.01 降到 0.002，成功消除了惩罚压制（ratio 从 -0.88 → -0.0008）。但压低惩罚后，agent 终于敢于接近目标区域，结果 soft_landing_proxy 的二值条件大量触发，暴露出 proxy 信号过强的问题。

**本轮行动**：不动 penalty 和 progress，只改 soft_landing_proxy 的数学形态——从二值条件改为连续乘积，提供梯度并自然降低均值。

参考 `bounded_continuous_proxy` 技法：用 `max(0, 1 - x/threshold)` 形式的连续因子乘积替代硬 if 条件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── 1. 提取观察量 ──
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]

    vel_x, vel_y = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 2. 主学习信号：进度差奖励（本轮不动）──
    dist_old = (x ** 2 + y ** 2) ** 0.5
    dist_new = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_old - dist_new

    # ── 3. 轻量稳定约束（本轮不动）──
    stability_penalty = -0.002 * (abs(vel_x) + abs(vel_y)) \
                        -0.002 * abs(angle) \
                        -0.002 * abs(angular_vel)

    # ── 4. 连续软着陆引导信号（形态改动：二值 → 连续乘积）──
    # 原因：二值条件 "near and slow and upright and legs_down → 0.5"
    # 导致 agent 在阈值边界无梯度、hovering exploit（nonzero_rate=51.5%，ratio=87x progress）
    # 改为连续因子乘积，每个因子用 max(0, 1 - x/D) 形式提供稠密梯度
    speed = (vel_x ** 2 + vel_y ** 2) ** 0.5

    prox_factor = max(0.0, 1.0 - dist_new / 0.3)       # dist=0→1, dist≥0.3→0
    speed_factor = max(0.0, 1.0 - speed / 0.5)          # speed=0→1, speed≥0.5→0
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.3)    # angle=0→1, |angle|≥0.3→0
    leg_factor = 0.5 * (left_contact + right_contact)   # 两腿→1, 单腿→0.5, 无→0

    # 乘积确保"同时满足"约束，系数 0.5 为完美姿态时的最大单步奖励
    soft_landing_continuous = 0.5 * prox_factor * speed_factor * angle_factor * leg_factor

    # ── 组合总奖励 ──
    total_reward = progress + stability_penalty + soft_landing_continuous

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_continuous": soft_landing_continuous,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

### 预期效果

- **soft_landing_continuous 均值将显著下降**：四因子乘积天然小，agent 不再能轻松收割 0.5/步。预期 mean 降到 0.01–0.05 区间，ratio_to_progress 从 87x 降到 ~3–15x。
- **梯度改善**：二值条件在 threshold 处梯度为零，连续因子 `max(0, 1-x/D)` 在 [0, D] 区间内提供恒定梯度，引导 agent 持续优化姿态。
- **防 hovering**：乘积形式意味着哪怕三个条件完美、一个条件勉强，奖励也被压低——agent 必须在所有维度上同时优化，而非刚好踩线。
