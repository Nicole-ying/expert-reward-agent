# 设计理由

## 信号覆盖审计

1.  **终止模式分析**：本轮 `terminated=0/20, truncated=20/20`，全部因超时（1000步）结束，说明 agent 未触发任何终止条件，但也没有完成任务。从 `contact_landing_reward` 高触发率（61%）和巨大累积值（2969.4）可以推断，agent 一直在目标区域附近活动，频繁与地面发生接触，但未能达到足够的稳定状态来触发环境终止（可能是速度/角度必须同时低于某个阈值并保持一段时间）。

2.  **观测使用扫描**：`compute_reward` 使用了所有8个观测维度（x, y, vx, vy, angle, ang_vel, left_contact, right_contact）。没有遗漏的传感器。

3.  **信号缺口判断**：**信号齐全但校准问题**。agent 的徘徊行为是由于 `contact_landing_reward` 对“近似着陆”状态过于慷慨，给了 agent 巨大的正向奖励，使得它满足于在目标点附近反复接触而非追求极致稳定，从而导致 episode 无法收敛到终止状态。`progress_gated` 因距离变化趋于零而失效，横向和角速度惩罚过小无法影响行为。

## 行为诊断

- **agent 在做什么？** 慢速徘徊/刷分 exploit。它在目标区域保持小角度、低高度、频繁接触，从 `contact_landing_reward` 中获得巨量奖励（每步平均约 3.0），但速度和角度尚未小到触发环境成功终止。它正在利用当前手指导的宽容阈值。
- **干预哪个目标？** 调整 `contact_landing_reward`，使其对“完美”状态更具区分力，驱散徘徊。
- **方向评估**：当前骨架已连续多轮优化并刷新 best score，未出现连续 3 轮 `❌`，因此保留现有框架，进行 Level 2 结构微调。

## 干预层级与数学变换

**Level 2 — 结构变换**：`contact_landing_reward` 当前的几何平均形式 (power 0.25) 对单个因子的不完美有很强的“平滑”作用（四个 0.5 的因子几何平均后仍有 0.5）。这使得 agent 在各项指标“差不多”时就能获得不菲的回报，缺乏追求极致稳定的动力。

**变换**：将幂次从 0.25 提高为 0.5。这使得奖励曲面更陡峭，对速度、角度、位置偏差的惩罚更明显，迫使 agent 为了获得相同量级的累计奖励必须进入更完美的状态（速度更低、角度更小、位置更居中），从而增加触发环境成功终止的概率。

**系数校准**：
- 理想情况（各因子 0.9）新奖励 ≈ 5 × (0.9⁴)^(0.5) ≈ 4.05 / step。
- 普通徘徊状态（各因子 0.5）新奖励 ≈ 5 × (0.5⁴)^(0.5) ≈ 1.25 / step，较旧版（2.5 / step）显著降低，有效削弱“足够好”状态的吸引力。
- 惩罚组件 per-step 极低（<0.002），远小于新主信号，总惩罚负担安全。

# 修改代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # observation indices: 0:x, 1:y, 2:vx, 3:vy, 4:angle, 5:ang_vel, 6:left_contact, 7:right_contact
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_lc = next_obs[6]  # left contact
    n_rc = next_obs[7]  # right contact

    # ---------- 1. progress towards origin (distance decrease) ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new
    w_progress = 30.0

    # ---------- 2. attitude gate: suppress progress when angle is dangerous ----------
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (12.0 * (abs_angle - 0.30)) / 
                              (1.0 + 2.718281828 ** (12.0 * (abs_angle - 0.30))))

    # ---------- 3. lateral position penalty (encourage centering) ----------
    w_lat_pos = 0.08
    lateral_pos_penalty = -w_lat_pos * (nx ** 2)

    # ---------- 4. angular velocity penalty (smooth rotation) ----------
    w_angvel = 0.05
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. contact-based landing proxy (sharper exponent) ----------
    mean_contact = (n_lc + n_rc) / 2.0

    k_y = 10.0
    k_vy = 8.0
    k_ang = 15.0
    f_y   = 1.0 / (1.0 + k_y   * abs(ny))
    f_vy  = 1.0 / (1.0 + k_vy  * abs(nvy))
    f_ang = 1.0 / (1.0 + k_ang * abs(n_angle))

    # 仅修改了这一行：将指数从 0.25 提升到 0.5，使奖励曲面对非完美状态更陡峭
    contact_landing_factor = (mean_contact * f_y * f_vy * f_ang) ** 0.5
    w_contact_land = 5.0
    contact_landing_reward = w_contact_land * contact_landing_factor

    # ---------- combine ----------
    total_reward = (w_progress * progress * angle_gate
                    + lateral_pos_penalty
                    + angvel_penalty
                    + contact_landing_reward)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_pos_penalty": lateral_pos_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_landing_reward": contact_landing_reward
    }
    return float(total_reward), components
```

# 诊断摘要