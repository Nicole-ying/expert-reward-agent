# 设计理由

**第 0 步审计结论**：所有 8 个观测维度均已使用，但 soft_landing 的 proximity gate 从未被越过（active_rate 0%），agent 在远处徘徊满 1000 步结束。approach 组件（distance progress + velocity alignment）虽然提供了大量正奖励（episode_sum_mean=537），但该正奖励在任何距离下只要距离稍减就能获取，导致 agent 学会在小范围抖动“刷分”而不向原点收敛。信号齐全但校准错误——progress 信号在远处缺乏距离感知，需要改为直接吸引向原点的有界正奖励。

**修改组件**：将 `approach` 替换为 `proximity_reward`。原 approach 使用 `dist_prev - dist_next`（改善量）加上速度方向对齐，在远离原点时同样可以轻易获得正奖励。新组件改用 `1.0 / (1.0 + dist_next)`，在远处给出约 0.1–0.3 的小正向奖励，越靠近原点奖励越大（最大 1.0），直接激励 agent 缩减距离，从而最终触发 soft_landing gate。同时保留 orientation_penalty 与 soft_landing 的结构不变，等待 agent 接近原点后 soft_landing 自然苏醒。

**数学形式**：单组件有界正奖励，`reward = w_proximity / (1.0 + dist_next)`。选择 `w_proximity = 1.0`，使 per-step 奖励在 0.0–1.0 之间，总 episode 潜在正奖励上限约 1000，不会产生极端值支配。经验上该幅度与之前成功骨架（第 1 轮 progress_delta）的量级相当。

**系数校准**：无额外惩罚负担。soft_landing 仍为 2.0 倍原值，但仅在 `dist_next < 0.3` 时生效，远离原点时为零。orientation_penalty 保持为极小（原系数不变），只起姿态微调作用。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Replace unbounded progress+alignment with direct proximity attraction.
    Agent receives positive reward inversely proportional to distance from origin,
    which monotonically increases as it approaches the center landing platform.
    Soft-landing and orientation penalty remain unchanged.
    """

    # ── Unpack observations ──────────────────────────────────────────
    px1, py1 = next_obs[0], next_obs[1]  # current position
    vx1, vy1 = next_obs[2], next_obs[3]  # current velocity
    angle1  = next_obs[4]                # body angle
    angvel1 = next_obs[5]                # angular velocity
    left_leg  = next_obs[6]              # left contact
    right_leg = next_obs[7]              # right contact

    # ── Derived signals ─────────────────────────────────────────────
    dist_next = (px1**2 + py1**2) ** 0.5
    speed = (vx1**2 + vy1**2) ** 0.5

    # ── 1. Proximity reward (replaces approach) ─────────────────────
    # Bounded attraction: closer → larger reward (max 1.0 at origin)
    proximity_reward = 1.0 / (1.0 + dist_next)

    # ── 2. Orientation / stability penalties (unchanged) ────────────
    angle_penalty  = -0.01 * (angle1 ** 2)
    angvel_penalty = -0.005 * (angvel1 ** 2)
    orientation_penalty = angle_penalty + angvel_penalty

    # ── 3. Soft landing guidance (unchanged) ─────────────────────────
    proximity_threshold = 0.3
    if dist_next < proximity_threshold:
        contact_factor = (left_leg + right_leg) / 2.0
        speed_factor   = 1.0 / (1.0 + 10.0 * speed)
        angle_factor   = 1.0 / (1.0 + 5.0 * (angle1**2))
        soft_landing   = contact_factor * speed_factor * angle_factor
    else:
        soft_landing = 0.0

    # ── Combine ──────────────────────────────────────────────────────
    total_reward = (
        1.0 * proximity_reward
        + 1.0 * orientation_penalty
        + 2.0 * soft_landing
    )

    components = {
        "proximity_reward":    proximity_reward,
        "orientation_penalty": orientation_penalty,
        "soft_landing":        soft_landing,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: soft_landing 门控从未触发（proximity_threshold=0.3 未被越过），approach 在任意距离均给予正向 progress 奖励，诱使 agent 在远处原地抖动刷分而不趋近原点，所有观测已用但信号方向错误。
- **behavior**: agent 在远处小幅移动以持续获取 distance progress 正奖励，存活满 1000 步超时，从未接近着陆平台。
- **signal**: 前进信号缺失距离感知——需将“改善量”替换为“绝对距离吸引”，使靠近原点本身成为持续正激励。
- **level**: Level 2（结构变换：unbounded progress → dense proximity attraction）
- **hypothesis**: 直接奖励距离缩小（1/(1+dist)）在远处提供温和引导，在近处提供强力吸引，迫使 agent 向原点运动，最终穿过 proximity_threshold 激活 soft_landing，使成功着陆的奖励闭合回路。
- **risk**: 若初始位置远离原点且姿态控制不足以克服漂移，agent 可能仍无法靠近原点。但新组件在远处仍给予正向奖励（不会惩罚），保留了探索余地。