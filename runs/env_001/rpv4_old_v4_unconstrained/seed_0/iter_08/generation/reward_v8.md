Now I have a thorough picture. Let me compile the full diagnosis.

---

## 诊断报告

### Evidence

| 证据来源 | 关键数据 |
|---|---|
| **iter_7 score** | 83.19（目标200），terminated=8/20（仅40%成功着陆），len=766.6（过长） |
| **iter_7 landing_bonus** | magnitude_share=**81.0%**，ep_sum=261.7，active_rate=5.1% — 尽管已从v4的97.4%降低，仍严重主导 |
| **iter_7 checkpoint landing_bonus** | cp60%=303 → cp80%=**669.5** → cp90%=150.8 → cp100%=218.9：**典型exploit-collapse模式** |
| **iter_7 score trend** | cp60%=114.3 → cp80%=**130.4（峰值）** → cp100%=107.7：训练后期退化 |
| **iter_4 (best)** | score=120.84，landing_bonus=2627.5（97.4%），同样exploit-collapse（cp90%=5387→cp100%=665） |
| **iter_5** | vel_damping扩大门控(sigma=1.5, coeff=-4.0)→agent从未触地，landing/contact均为0 |
| **iter_6** | transition detection (+200/+5)，score=39，landing仍78.6%主导 |

### Behavior Diagnosis

**核心问题：landing_bonus 的 PRODUCT 公式制造了「悬崖式」奖励地形，导致 exploit → collapse 循环。**

v7 使用 `landing_quality = dist_quality × angle_quality × speed_quality × contact_quality`。四个因子相乘意味着：只要任一因子接近零（如角度稍大、速度稍快），整体奖励直接归零。这造成：

1. **探索极度困难**：飞行中偶然触发一次高质量着陆的概率极低，agent 大部分时间收到零 landing 信号
2. **一旦偶然成功 → 过度拟合**：agent 记住了恰好触发高奖励的那组精确条件，在 cp80% 处 landing_bonus 飙升到 669.5
3. **继续探索 → 策略崩溃**：稍微偏离那个「幸运点」，奖励骤降为零，策略迅速退化（cp90%=150.8，cp100%=218.9）
4. **结果**：最终只有 40% 的 episode 成功着陆（terminated=8/20），60% 超时截断（len=766）

**v4 的 binary 版本有同样的问题**，只是表现形式不同：binary 阈值触发后 agent 学会了刚好满足条件，然后在继续训练中崩溃。

**v5 的教训**：扩大 vel_damping 门控会杀死下降动力——agent 害怕任何速度，永远不触地。

**v6 的教训**：transition detection（+200 一次性/+5 持续）方向正确，但 +200 仍太大，+5/step 持续奖励仍会累积。

### 干预策略

三个关键修改：

1. **PRODUCT → SUM**：`landing_quality = dist + angle + speed + contact`。每个因子独立贡献，改善任一维度都能获得增量奖励——形成平滑梯度，消除「悬崖」。

2. **一次性 touchdown bonus 取代 per-step 累积**：在「首次触地瞬间」（air→ground 转移）给予一笔按质量缩放的奖励（max 80），之后无持续奖励。这从根本上消除了 per-step 累积导致的 dominate 问题。

3. **降低 contact_reward**：从 0.5/foot 降到 0.2/foot，减少地面停驻期间的累积，仅保留 ground-discovery 梯度。

预期效果：landing_bonus magnitude_share 从 81% 降至 30-40%，agent 获得更平滑的学习信号，terminated 率回升。

---

## v8 奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v8: Sum-based touchdown quality + one-time bonus. Eliminates per-step landing accumulation.

    Root cause of v4/v6/v7 failures:
    - Per-step landing rewards (binary 50/step or product 12/step) accumulate while agent
      sits on ground, inevitably dominating the reward landscape.
    - Product formulation (v7) creates multiplicative cliffs: one bad factor → reward=0.
    - This causes the classic exploit-collapse pattern at cp80% in training.

    Fixes:
    1. PRODUCT → SUM: landing_quality = dist + angle + speed + contact_quality.
       Each factor contributes independently → smooth gradient, no cliffs.
    2. ONE-TIME touchdown bonus: reward is given only at the moment of first ground
       contact (air→ground transition), proportional to landing quality.
       Max ~80, no per-step sustain → cannot dominate through accumulation.
    3. contact_reward reduced to 0.2/foot — retains ground-discovery gradient without
       excessive accumulation.
    4. All other components at v4/v7-proven scales.
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    px = obs[0]
    py = obs[1]
    p_left = obs[6]
    p_right = obs[7]

    # ── Derived quantities ──
    dist = (x ** 2 + y ** 2) ** 0.5
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # ── 1. Progress: delta-distance (v7 scale, proven navigation signal) ──
    progress_reward = 2.0 * (prev_dist - dist)

    # ── 2. Orientation penalty (v7 scale: stable, always active) ──
    orientation_penalty = -0.1 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 3. Velocity damping near pad (v4/v7 scale: narrow gate, modest coefficient) ──
    proximity_gate = 2.718281828 ** (- (dist ** 2) / 0.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * proximity_gate * excess_speed

    # ── 4. Contact reward: modest ground-discovery incentive (reduced from 0.5) ──
    contact_reward = 0.2 * (left_contact + right_contact)

    # ── 5. Landing bonus: ONE-TIME touchdown quality (SUM-based, no per-step sustain) ──
    # Detect air→ground transition
    was_airborne = (p_left < 0.5 and p_right < 0.5)
    touching_now = (left_contact > 0.5 or right_contact > 0.5)
    just_touched_down = was_airborne and touching_now

    if just_touched_down:
        # Each quality factor: 0 (worst) to 1 (perfect), linear decay
        dist_quality = max(0.0, 1.0 - dist / 0.5)           # 1 at center, 0 at dist>=0.5
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.4)    # 1 when upright, 0 at |angle|>=0.4
        speed_quality = max(0.0, 1.0 - speed / 0.5)         # 1 when still, 0 at speed>=0.5
        contact_quality = (left_contact + right_contact) / 2.0  # 0, 0.5, or 1.0

        # SUM: each factor contributes independently — smooth gradient, no cliffs
        td_quality = dist_quality + angle_quality + speed_quality + contact_quality
        # Max td_quality = 4.0 → max bonus = 80.0 (one-time, no accumulation)
        landing_bonus = 20.0 * td_quality
    else:
        landing_bonus = 0.0

    # ── 6. Action efficiency (v4/v7 scale) ──
    if action == 0:
        action_reward = 0.0
    elif action == 2:          # main engine — highest cost
        action_reward = -0.04
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.02

    # ── Aggregate ──
    total_reward = (
        progress_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + landing_bonus
        + action_reward
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "action_reward": action_reward,
    }
    return float(total_reward), components
```