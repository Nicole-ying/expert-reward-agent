# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation (state after action)
    x_pos = next_obs[0]          # horizontal distance to target pad
    y_pos = next_obs[1]          # vertical distance (height) to target pad
    x_vel = next_obs[2]          # horizontal velocity
    y_vel = next_obs[3]          # vertical velocity
    body_angle = next_obs[4]     # tilt angle
    ang_vel = next_obs[5]        # angular velocity
    left_contact = next_obs[6]   # left support leg contact (0/1)
    right_contact = next_obs[7]  # right support leg contact (0/1)

    # 1. Main learning signal: distance‑based reward
    # Encourage getting close to the center of the target pad.
    distance = (x_pos**2 + y_pos**2) ** 0.5
    r_dist = 1.0 / (1.0 + distance)        # in (0, 1], higher when closer

    # 2. Stability / smoothness penalty
    # Penalize high speeds, large tilt, and high angular velocity to promote gentle flight and soft landing.
    penalty = 0.01 * (abs(x_vel) + abs(y_vel)) + 0.1 * abs(body_angle) + 0.05 * abs(ang_vel)
    r_stability = -penalty

    # 3. Soft landing proxy (task‑completion approximation)
    # Give a one‑time bonus when the lander touches down softly on both legs near the pad center.
    landing_bonus = 0.0
    if left_contact == 1.0 and right_contact == 1.0:
        if distance < 0.2 and abs(x_vel) < 0.3 and abs(y_vel) < 0.3 and abs(body_angle) < 0.2:
            landing_bonus = 10.0

    total_reward = r_dist + r_stability + landing_bonus

    components = {
        "distance_reward": r_dist,
        "stability_penalty": r_stability,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

| 组件 | 角色 | 数学形式 |
|------|------|----------|
| `distance_reward` | 主学习信号 | `1/(1+√(x_pos²+y_pos²))` – 连续引导 agent 靠近目标垫中心，越近奖励越高（最大值趋近 1）。 |
| `stability_penalty` | 稳定/安全约束 | 负加权和：`-0.01·(|x_vel|+|y_vel|) - 0.1·|angle| - 0.05·|ang_vel|` – 抑制高速、大倾斜和高角速度，鼓励平稳运动和软着陆。 |
| `landing_bonus` | 任务完成近似信号（proxy） | 当“双腿接触 + 距离 < 0.2 + 速度 < 0.3 + 倾角 < 0.2”时，一次性奖励 `+10`。用多个连续条件组合，杜绝单独利用接触信号。 |

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡片明确声明 **explicit_success_flag_available = false**，**explicit_failure_flag_available = false**，且 `info` 为空字典。
- 任何直接读取 `info['success']` 或推断硬终止原因的奖励都不可靠，因此在 v1 中完全禁用。

## 后续迭代预留

- **energy_penalty**：当前 agent 应先学会飞到并稳定着陆，之后再对主/姿态引擎的不必要使用进行惩罚，防止策略在学会基本任务后仍然浪费推力。
- **time_penalty**：如果 agent 到达目标后长时间盘旋而不触发 `landing_bonus`，可在后续加入轻度时间惩罚。
- **gated_reward / dynamic_curriculum**：复杂门控与课程学习默认留到 v2+，v1 阶段保持信号简单明了。

## 训练后应观察的 failure mode

1. **目标附近悬停不落地**：若 agent 学会以极小速度悬停在 (0,0) 上方获得高 `distance_reward`，但不主动伸出支撑腿或降低高度，可能不会触发 `landing_bonus`。应检查 `landing_bonus` 的触发频率；如果很少出现，需调整距离/速度/倾角阈值，或考虑增加温和的“接近地面”信号（需避免接触奖励 hacking）。
2. **利用快速触地/弹跳**：若 agent 撞地后双腿瞬间接触获得 `landing_bonus`，紧接着弹飞，在单 episode 中可能多次得分。观察 `landing_bonus` 步均频率与着陆稳定性；必要时收束阈值或增加“接触后仍需保持 X 步稳定”的条件（后续迭代考虑）。
3. **稳定性惩罚压制探索**：若 `stability_penalty` 权重过高，agent 可能不敢使用主引擎，导致坠落或无法抵达目标。可通过分析各组件步均值判断，如果惩罚项长期占主导，需降低相应系数。
4. **距离奖励与稳定性惩罚的平衡**：两者共控飞行风格。若 agent 始终以极高速度冲向目标并 crash，说明稳定性惩罚太弱；若移动过于缓慢，说明惩罚太重。训练曲线应显示兼顾速度和安全的策略。
