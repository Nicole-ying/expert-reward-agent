# Response Record

# 设计理由

## 第0步审计
- **终止模式**：18/20 terminated，仅 2/20 truncated，len=394。终止主要是身体摔倒（非早期快速失败，early_terminal 仅 4/20）。evidence：score 分布 [-109, -4.3]，说明部分 episode 走到较远但摔倒或低分到达终点。
- **观测使用扫描**：关键观测（hull_angle, hull_angular_vel, horizontal_speed, vertical_speed, leg_contacts）均已使用。关节角度/速度（obs[4:11]）和 lidar（obs[14:23]）未使用。关节力矩相关的能量效率是次生需求，不影响当前主目标。lidar 可用于预警但非当前卡点。
- **信号缺口判断**：**信号齐全但校准问题**。核心信号均已覆盖，问题在于 speed↔stability 的 trade-off 过于保守。
- **僵尸组件**：air_penalty active_rate=1.3%（双脚同时离地极少）→ 正常现象，不是僵尸。其余组件 active_rate 正常。

## 行为诊断
agent 在做什么：**慢速保守行走**。progress_reward per-step ≈ 0.044（从 episode_sum_mean=17.20 / len≈394 推算）。反推平均 horizontal_speed ≈ 0.22，posture_gate 均值 0.664（对应倾角 ≈ 0.10 rad）。agent 学会了维持小倾角换取 posture_gate 的高值，但速度被严重压制，累积 score 无法突破。

历史证据：iter 2 将 posture_gate 乘到 progress_reward 上，预期「迫使 agent 在倾斜时自动减速/纠姿」，实际 len 增加但 score 不变（-61.55 → -61.57），预判 ❌——agent 只是降低了速度而非提升稳健性。

## 干预目标
**progress_reward 组件**：从线性 `0.3 * v * gate` 改为凸化 `0.6 * v² * gate`，强化对高速度的边际激励，打破低速徘徊均衡。

## Formula switching 依据
指南明确：「线性正奖励 `w * signal`，score 停滞在低水平，signal 正值但偏小 → dense_state_signal (凸化)，改用 `signal**2`」。当前 horizontal_speed 约 0.22，线性奖励无法激励突破；改用二次型后，速度从 0.22 提升到 0.5 可使 per-step 奖励从 ~0.044 提升到 ~0.099（×2.25），形成突破性激励。

## 系数校准
- 在 v=0.22, gate=0.66 时：`0.6 × 0.0484 × 0.66 ≈ 0.019` —— 初期略低于当前的 0.044，agent 会因奖励下降而被迫加速。
- 在 v=0.4, gate=0.66 时：`0.6 × 0.16 × 0.66 ≈ 0.063` —— 超过当前水平。
- 在 v=0.6, gate=0.50 时（速度提升可能导致倾角增大）：`0.6 × 0.36 × 0.50 ≈ 0.108` —— 强烈正向激励。
- 惩罚负担合计 < 0.001 per-step，远低于主信号的 0.3× 阈值 ✓

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- 主学习信号：水平前进速度² × 姿态门控 -----
    horizontal_speed = obs[2]
    hull_angle = obs[0]

    # posture_gate: 连续衰减因子，倾角越大前进奖励越小
    posture_gate = 1.0 / (1.0 + 5.0 * abs(hull_angle))
    # 凸化速度奖励：低速时激励弱，高速时激励强，鼓励加速突破
    progress_reward = 0.6 * (horizontal_speed ** 2) * posture_gate

    # ----- 稳定/安全约束 -----
    # 1. 身体角速度惩罚 (quadratic，持续抑制旋转)
    hull_angular_vel = obs[1]
    angular_penalty = -0.06 * (hull_angular_vel ** 2)

    # 2. 垂直速度异常惩罚 (quadratic，弱抑制弹跳)
    vertical_speed = obs[3]
    vertical_penalty = -0.05 * (vertical_speed ** 2)

    # 3. 足部浮空惩罚：双脚同时离地常伴随跳跃或即将摔倒
    leg1_contact = obs[12]
    leg2_contact = obs[13]
    both_airborne = 1.0 if (leg1_contact + leg2_contact == 0.0) else 0.0
    air_penalty = -0.015 * both_airborne

    # ----- 汇总奖励 -----
    total_reward = progress_reward + angular_penalty + vertical_penalty + air_penalty

    components = {
        "progress_reward": progress_reward,
        "posture_gate": posture_gate,
        "angular_penalty": angular_penalty,
        "vertical_penalty": vertical_penalty,
        "air_penalty": air_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号齐全，speed-stability trade-off 过度保守，前进激励线性化导致低速徘徊
- **behavior**: agent 低速小倾角行走，以牺牲速度为代价维持 posture_gate 高值，score 停滞
- **signal**: horizontal_speed 线性奖励边际激励恒定，缺乏突破低速均衡的动力；已使用的信号覆盖完整
- **level**: Level 2
- **hypothesis**: 凸化速度奖励使加速的边际收益递增，agent 将被迫从低速保守策略中跳出，在速度和稳定性之间找到更高产出的平衡点，score 应显著提升
- **risk**: 速度提升可能导致摔倒率上升（倾角增大 → posture_gate 衰减），凸化奖励可能不足以完全抵消姿势惩罚的抑制力；若下一轮 len 下降或 terminated 率恶化，需调整 posture_gate 的分母系数（5→2）降低对正常倾角的惩罚
