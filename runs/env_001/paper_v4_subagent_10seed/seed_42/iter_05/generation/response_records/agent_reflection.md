# Response Record

# 设计理由

**修改的组件**：将原先的 `safe_progress`（gate‑constrained distance progress）替换为新的 `approach` 组件。新组件包含两项：

1. **无门控的 distance progress** — `max(0, dist_prev − dist_next)`，让下降距离的每一步都获得正向奖励，不再被 speed gate 束缚。
2. **velocity alignment** — 速度向量与指向目标方向的余弦相似度（仅取正值），奖励「朝向目标移动」的每一步。该项有界（0~1），不会因远距离而爆炸。

**为什么这样改**：
- 本轮 `safe_progress` 的 episode_sum_mean 只有 1.38（2% 份额），说明 speed gate 过度抑制了接近奖励，导致整个密集前进信号的强度失效。agent 几乎完全依赖稀软的 `soft_landing`（active_rate 仅 12.5%）学习，造成了长度仍高达 812 步、得分难以突破 best（163.33）的局面。
- 历史记忆 iter1（简单的 progress_delta 无 gate）拿到了 best，进一步证实 speed gate 是负面因素。
- velocity alignment 补充了「方向正确性」的密集引导，使 agent 在远距离时不受速度大小的惩罚，但仍能区分靠近与远离，避免盲目加速。

**系数校准**：
- `w_progress = 2.0`：假设 agent 平均每步距离缩短 0.01，贡献 0.02 per step。在整个 episode（约 400 步）约为 8，远低于当前主导的 `soft_landing`（69），不会喧宾夺主。
- `w_alignment = 1.0`：alignment 均值约 0.3–0.5，每步贡献约 0.3 – 0.5，总计 ≈ 120，略高于 soft_landing 但仍在可控范围。若后续发现主导性太强可下调。
- 保留了 `soft_landing`（×2.0）和 `orientation_penalty`（弱值）不变，不增加惩罚负担。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Approach: replace safe_progress with unbounded distance progress + velocity alignment.
    Soft-landing and weak orientation penalty unchanged.
    """

    # ── Unpack observations ──────────────────────────────────────────
    px0, py0 = obs[0], obs[1]          # last position
    px1, py1 = next_obs[0], next_obs[1]  # current position
    vx1, vy1 = next_obs[2], next_obs[3]  # current velocity
    angle1  = next_obs[4]                # body angle
    angvel1 = next_obs[5]                # angular velocity
    left_leg  = next_obs[6]              # left contact
    right_leg = next_obs[7]              # right contact

    # ── Derived signals ─────────────────────────────────────────────
    dist_prev = (px0**2 + py0**2) ** 0.5
    dist_next = (px1**2 + py1**2) ** 0.5
    speed_sq  = vx1**2 + vy1**2
    speed     = speed_sq ** 0.5

    # ── 1. Approach (distance progress + velocity alignment) ─────────
    # progress: positive when moving toward origin
    raw_progress = dist_prev - dist_next
    progress     = max(0.0, raw_progress)

    # alignment: cosine similarity between velocity and direction-to-target
    if dist_next > 1e-6 and speed > 1e-6:
        # direction to target is (-px1, -py1)
        dot = vx1 * (-px1) + vy1 * (-py1)
        alignment = dot / (dist_next * speed)
    else:
        alignment = 0.0      # at origin or stationary
    alignment = max(0.0, alignment)   # only reward approaching motion

    w_progress  = 2.0
    w_alignment = 1.0
    approach    = w_progress * progress + w_alignment * alignment

    # ── 2. Orientation / stability penalties (unchanged, weak) ──────
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
        1.0 * approach
        + 1.0 * orientation_penalty
        + 2.0 * soft_landing
    )

    components = {
        "approach":            approach,
        "orientation_penalty": orientation_penalty,
        "soft_landing":        soft_landing,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测已被使用，不存在信号缺失；问题在于密集进度信号被 speed gate 过度压制，导致 agent 几乎只靠稀疏 soft_landing 学习，长度居高不下。
- **behavior**: agent 学会了缓慢接近并最终 soft landing（terminated 16/20，len≈813），但缺乏快速接近的动力。
- **signal**: safe_progress 严重欠信号（ep_mean=1.38 vs soft_landing=69.2），密集前进奖励消失，agent 只被终点 sparseness 引导。
- **level**: Level 2 — 结构变换（移除 gate，加入 velocity alignment 丰富密集信号）
- **hypothesis**: 密集的 distance progress + alignment 将使每步都有引导，agent 会更快趋近目标，缩短 episode 长度并突破当前 score 瓶颈。
- **risk**: 移除 speed gate 后可能出现高速冲向目标导致的 overshoot 或 crash（若姿态不稳定）；当前弱 stabilization penalty 可能不足，下一轮若 terminated 失败率升高需加强姿态约束或引入 soft speed limiter。
