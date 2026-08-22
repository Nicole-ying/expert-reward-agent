现在让我全面分析 iter_3 的问题和整个历史脉络。

---

## 诊断报告

### 1. 历史轨迹回顾

| iter | 核心变化 | score | len | 终止 | 行为 |
|------|---------|-------|-----|------|------|
| 1 | contact_reward (死) + vel_damping (死) | -159 | 303 | 20/20 | 快速坠毁 |
| 2 | +soft_landing_proxy (连续状态值奖励) | -100 | **1000** | 0/20 | **悬停 exploit**：在着陆区盘旋 1000 步吃奖励，不真正着陆 |
| 3 | 换为 landing_bonus (事件型) | -113 | 69 | 20/20 | **快速坠毁**：稀疏信号无法引导学习 |

### 2. Evidence（iter_3 关键数据）

| 信号 | 数值 | 诊断 |
|------|------|------|
| **vel_damping** | ep_sum=-19.98, magnitude_share=**53.9%**, active=100% | 最大的负信号，且 checkpoint 全程平坦在 -20，**RL 完全无法优化它** |
| **landing_bonus** | ep_sum=12.44, active_rate=**2.2%** | 几乎不触发——agent 活不到接触地面就 crash 了 |
| **progress_reward** | ep_sum=3.36, active=100% | 唯一的连续正向引导，但太弱，只占 magnitude 9.4% |
| orientation_penalty | 从 cp10% 的 -2.6 改善到 cp100% 的 -0.8 | **唯一被成功优化的组件**——agent 学会了控姿态 |
| **early_terminal** | 19/20 (<150步且score<-50) | agent 几乎全在早期坠毁 |
| score trend | cp50%=-104 → cp80%=-98.6(peak) → cp100%=-111.2 | 中后期曾进步但又退化 |

### 3. Behavior Diagnosis

**当前策略学到的是「飞向目标→失控坠毁」**。agent 在 progress_reward 驱动下靠近着陆垫，但缺乏足够的减速引导。vel_damping 占主导但它是无差别的速度惩罚（水平+垂直一起罚），且全程平坦说明 agent 无法通过改变行为来减少它。landing_bonus 触发率仅 2.2%，意味着 agent 极少活着接触到地面——更别说软着陆。

**核心矛盾**：iter_2 有连续引导但被悬停 exploit；iter_3 消除了 exploit 但也消灭了引导信号。需要在「可 exploit 的连续状态奖励」和「太稀疏无法学习的事件奖励」之间找到第三条路。

### 4. 干预方案

**语义方向**：将 vel_damping 从「无差别制动」改造为「方向性着陆引导」——区分水平漂移（始终危险）和垂直下降（着陆必需），同时大幅增强 progress_reward 提供足够的接近动力。这是一次语义修改（「教会 agent 快速接近 + 定向减速」）。

**修改层级**：L2（改 vel_damping 数学形态）+ L1（调 progress/landing 权重联动）

**具体变化**：
- **vel_damping (L2)**：分离水平速度和垂直速度。水平速度全罚（漂移危险），垂直速度设置 0.8 死区（允许受控下降），整体权重从 0.3 降至 0.2
- **progress_reward (L1)**：权重 3.0 → 12.0，提供足够强的连续接近引导（潜力函数形式，不会被悬停 exploit）
- **landing_bonus (L1)**：权重 60 → 80，增大成功着陆的正反馈强度

---

## 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v4: Directional velocity damping — penalise horizontal drift fully,
        allow controlled vertical descent (deadzone 0.8);
        strongly boost progress weight to replace lost continuous guidance;
        up landing bonus for stronger touchdown signal.
    """
    # ── Extract signals ──────────────────────────────────────────────
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # ── 1. Progress reward (potential-based: bounded total, no hover exploit) ─
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # >0 when approaching
    w_progress = 12.0                                    # boosted from 3.0 → strong approach incentive
    progress_reward = w_progress * progress

    # ── 2. Orientation penalty (unchanged) ───────────────────────────
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ── 3. Landing contact bonus (transition-based, unchanged structure) ─
    prev_contacts = (1.0 if obs[6] > 0.5 else 0.0) + (1.0 if obs[7] > 0.5 else 0.0)
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)
    new_contacts = max(0.0, curr_contacts - prev_contacts)   # 0, 1, or 2 newly touching

    speed = (vx ** 2 + vy ** 2) ** 0.5
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 1.5 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.5 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 80.0                                    # boosted from 60
    landing_bonus = w_landing * new_contacts * quality

    # ── 4. Directional velocity damping ──────────────────────────────
    # Distinguish: horizontal drift (always dangerous for landing on target)
    #              vs vertical descent (necessary, allow up to 0.8 deadzone)
    dist_thresh = 3.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 far away, 1 at origin

    horizontal_penalty = abs(vx)                         # full penalty on lateral drift
    vertical_penalty = max(0.0, abs(vy) - 0.8)           # deadzone: allow moderate descent

    w_vel_damp = 0.2                                     # reduced from 0.3
    vel_damping = -w_vel_damp * gate * (horizontal_penalty + vertical_penalty)

    # ── 5. Action efficiency (unchanged) ─────────────────────────────
    if action == 0:                                      # no engine
        action_penalty = 0.0
    elif action == 1 or action == 3:                     # orientation engines
        action_penalty = -0.05
    else:                                                # action == 2, main engine
        action_penalty = -0.2

    # ── Assemble ─────────────────────────────────────────────────────
    total_reward = (
        progress_reward +
        orientation_penalty +
        landing_bonus +
        vel_damping +
        action_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus,
        "vel_damping": vel_damping,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```