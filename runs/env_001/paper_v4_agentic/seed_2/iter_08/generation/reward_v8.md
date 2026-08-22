# 设计理由

## 信号覆盖审计（第 0 步结论）

**终止模式分析**：本轮 20/20 truncated（超时 1000 步），无 terminated 回合。len=1000 表明 agent 已学会不在空中摔死或飞出边界，能持续存活到超时。但 target score=200 vs current=145，差距 54.5 分，说明虽然活着，但着陆质量不够好——agent 在垫子上空反复调整但未达到"稳定停靠"的理想状态。

**观测使用扫描**：上一轮代码使用了全部 8 个观测维度（obs[0]~[7]），无遗漏信号。对应的x, y, vx, vy, angle, angular_vel, left_contact, right_contact 全部参与计算。

**僵尸组件检查**：无僵尸组件。A_progress_gated active_rate=100%，C_landing_steady active_rate=74.9%，两者都在工作中。

**信号缺口判断**：**信号齐全但校准问题**。问题不在于需要新观测，而在于现有组件的数学形态导致 reward 分布扭曲——C_landing_steady（episode_sum_mean=197.8）完全主导了总奖励，A_progress_gated（episode_sum_mean=5.63）沦为噪声。但 C_landing_steady 的触发条件过于苛刻（5 个 factor 乘积），导致 agent 在大部分时间获得的 reward guids 几乎为 0，只在恰好满足所有条件时获得大额奖励。这使得梯度稀疏，agent 难以朝更优方向精细调整。

## 诊断

**agent 在做什么？**：agent 已通过 A_progress_gated 学会了接近着陆垫，并通过 C_landing_steady 的因子学会了触发双接触。但 C_landing_steady 的 `speed_factor`（速度需 < 0.1）和 `angle_factor`（角度需 < 0.1）的阈值极其严苛——安全着陆通常不需要如此极端的精度。这导致 agent 在垫子上反复"跳针"却无法持续获得正向梯度，因为每当它接近但未完全满足这两个阈值时，乘积塌缩为 0。

**干预哪个目标？**：修复 C_landing_steady。这是占 97% 奖励但梯度稀疏的组件。问题不是它太强——而是它在"较好但不完美"的区域奖励为 0，梯度断裂。

**方向是否值得继续？**：iter 6-7 骨架运行良好（score 从 -116 跃升至 145），无连续 ❌。当前方向有效但需要校准。

## 层级与变换：Level 2 — 结构变换

**证据**：
- C_landing_steady 的 `speed_factor` 和 `angle_factor` 在"可接受但不够完美"的状态下塌缩为 0。agent 在 74.9% 的 step 中获得非零奖励，说明 25% 的 step 落在阈值外。
- 乘积形式 `dist_factor * contact_factor * speed_factor * angle_factor * angular_factor` 中任一因子为 0 则整体为 0。

**变换**：将 `speed_factor` 和 `angle_factor` 的阈值从极严的 0.1 放宽至合理范围（0.3 和 0.2），并使用更平缓的线性衰减而非锐利的 max(0, 1 - x/threshold)。这使 agent 在"较好但不完美"的区域获得部分信用而非完全归零。

同时，降低 C_landing_steady 的系数从 0.3 至 0.15，因为放宽阈值后 active_rate 将上升，每步贡献更频繁，需要降低单步系数以保持 per-step 尺度合理。

## 系数校准

- **主信号 per-step**：A_progress_gated 的 episode_sum_mean=5.63 / 1000 = 0.0056。C_landing_steady=197.84/1000=0.198。
- **设计目标**：放宽阈值后，C_landing_steady 的 active_rate 预计从 74.9% 升至约 85-90%。系数从 0.3 降至 0.15 使 episode 总量从 ~198 降至约 ~100（仍占主导但不过度）。这为目标分数 200 留有上升空间（A_progress 随姿态改善会自然增长）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- Extract next_obs signals ---
    x = next_obs[0]          # horizontal position relative to pad center
    y = next_obs[1]          # vertical height above pad
    vx = next_obs[2]         # horizontal velocity
    vy = next_obs[3]         # vertical velocity
    angle = next_obs[4]      # body angle (0 upright)
    angular_vel = next_obs[5]  # angular velocity
    left_contact = next_obs[6]  # left leg contact
    right_contact = next_obs[7] # right leg contact

    # --- Helper: distance from pad center (target) ---
    horizontal_dist = abs(x)
    distance_to_target = (horizontal_dist**2 + y**2) ** 0.5

    # --- Component A: main progress signal via potential-based shaping ---
    norm_distance = distance_to_target / 2.5
    angle_penalty = abs(angle) / 1.57
    potential = -(norm_distance + 0.3 * angle_penalty)

    prev_x = obs[0]
    prev_y = obs[1]
    prev_angle = obs[4]
    prev_horizontal_dist = abs(prev_x)
    prev_distance = (prev_horizontal_dist**2 + prev_y**2) ** 0.5
    prev_norm_distance = prev_distance / 2.5
    prev_angle_penalty = abs(prev_angle) / 1.57
    prev_potential = -(prev_norm_distance + 0.3 * prev_angle_penalty)

    potential_delta = potential - prev_potential

    # --- Component B: soft velocity health gate (unchanged) ---
    speed = (vx**2 + vy**2) ** 0.5
    safe_speed = 0.3 + 1.5 * distance_to_target
    overspeed_ratio = speed / (safe_speed + 1e-6)
    gate = 0.3 + 0.7 * (2.718281828 ** (-max(0.0, overspeed_ratio - 1.0)))

    scaled_progress = potential_delta * 10.0
    gated_progress = scaled_progress * gate

    # --- Component C: landing steady-state reward (REVISED thresholds) ---
    # Distance factor: linear decay, zero beyond 0.25 (slightly relaxed from 0.15)
    dist_factor = max(0.0, 1.0 - distance_to_target / 0.25)

    # Contact factor: requires BOTH legs on pad
    contact_factor = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0

    # Speed factor: relaxed from 0.1 to 0.3 threshold
    # Soft landing below 0.3 is acceptable; linear decay to 0 at 0.5
    if speed < 0.3:
        speed_factor = 1.0
    else:
        speed_factor = max(0.0, 1.0 - (speed - 0.3) / 0.2)

    # Angle factor: relaxed from 0.1 to 0.25 threshold
    # Near-upright below 0.25 rad (~14 deg) is good enough
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.25)

    # Angular velocity factor: relaxed from 0.5 to 0.8 threshold
    angular_factor = max(0.0, 1.0 - abs(angular_vel) / 0.8)

    # Product — any factor can zero out if far from target
    landing_factor = dist_factor * contact_factor * speed_factor * angle_factor * angular_factor
    C_landing = 0.15 * landing_factor  # reduced from 0.3 due to higher active_rate

    total_reward = gated_progress + C_landing

    components = {
        'A_progress_gated': gated_progress,
        'C_landing_steady': C_landing
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号齐全但校准问题——C_landing_steady 的 speed_factor 和 angle_factor 阈值过于严苛，导致在"可接受但不完美"区域奖励塌缩为 0，梯度断裂。
- **behavior**: agent 已学会接近垫子并触发双接触，但在垫子上反复调整无法稳定，因为速度和角度的微小偏差使奖励完全归零，缺乏精细梯度引导。
- **signal**: C_landing_steady 的主引导梯度因阈值锐利而稀疏；放宽阈值并提供线性衰减区能提供连续梯度信号。A_progress_gated 功能正常。
- **level**: Level 2
- **hypothesis**: 放宽 speed_factor 阈值（0.1→0.3/0.5）和 angle_factor 阈值（0.1→0.25）将使 agent 在"较好但不完美"区域获得部分信用，提供连续梯度引导它逐步降低速度和对齐角度，最终达到更高质量的着陆。降低系数至 0.15 防止因 active_rate 上升而过度放大。
- **risk**: 放宽阈值可能短期内让 agent 满足于"差不多"的着陆（得分暂时不升或微降），但中期的连续梯度应引导进一步精化。如果 speed_factor 阈值放宽过多导致 agent 着陆时速度过大，可能需要回调至 0.25。