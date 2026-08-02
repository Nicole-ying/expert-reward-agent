# Response Record

# 设计理由
当前 agent 已经从之前快速坠毁的状态走出，学会了飞到目标附近并减速悬停，因此 episode length 大幅增长（~956步），score 转正至 141。但 19/20 的 episode 以 truncation（超时）结束，说明 agent **会靠近目标但不会完成着陆**——它悬停在目标上方却不放下支撑腿接触目标垫。

审计发现 **接触信号（obs[6], obs[7]）完全未被使用**，这恰恰是区分“悬停”和“着陆成功”的关键。此外，angular_velocity_penalty 组件 active_rate 仅 0.1%，是僵尸组件，应删除。

本次 Level 2 修改：
1. **移除** 僵尸组件 `angular_velocity_penalty`。
2. **新增** `contact_stability` 组件：当机体有支撑接触且距离目标足够近时，给予与低速、直立程度成正比的奖励，激励 agent 在目标垫上保持稳定接触。
3. **新增** `inferred_success_bonus` 组件：当 `next_obs` 满足距离近、速度低、角度小、有接触四个条件时，推断为成功着陆并给予一次性大额奖励（+50），为达到 200 分目标提供强信号。

系数校准：`contact_stability` 每步最高 ~0.1，略高于当前 `soft_landing` 的 0.05，以拉动偏好。`inferred_success_bonus` 为 50，填补约 58 分的缺口。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D lander goal-reaching task.
    v5: removed zombie angular_velocity_penalty;
        added contact_stability and inferred_success_bonus to teach touchdown.
    """
    # ---------- constants ----------
    PROGRESS_WEIGHT = 1.0
    LANDING_WEIGHT = 0.05
    ANGLE_PENALTY_WEIGHT = 0.01

    CONTACT_WEIGHT = 0.1
    PROXIMITY_THRESHOLD = 0.5
    SUCCESS_DIST_THRESHOLD = 0.3
    SUCCESS_SPEED_THRESHOLD = 0.3
    SUCCESS_ANGLE_THRESHOLD = 0.2
    SUCCESS_BONUS = 50.0

    # ---------- unpack observations ----------
    x_o, y_o, x_v_o, y_v_o, angle_o, angvel_o, left_o, right_o = tuple(obs)
    x_n, y_n, x_v_n, y_v_n, angle_n, angvel_n, left_n, right_n = tuple(next_obs)

    # ---------- 1) progress to target ----------
    R_obs = (x_o ** 2 + y_o ** 2) ** 0.5
    R_next = (x_n ** 2 + y_n ** 2) ** 0.5
    progress_reward = PROGRESS_WEIGHT * (R_obs - R_next)

    # ---------- 2) soft landing incentive (continuous slowdown) ----------
    proximity = max(0.0, 1.0 - R_next / PROXIMITY_THRESHOLD)
    speed = (x_v_n ** 2 + y_v_n ** 2) ** 0.5
    speed_bonus = 1.0 / (1.0 + speed)
    soft_landing = LANDING_WEIGHT * proximity * speed_bonus

    # ---------- 3) light angular penalty ----------
    angle_penalty = -ANGLE_PENALTY_WEIGHT * (angle_n ** 2)

    # ---------- 4) contact stability reward ----------
    contact_flag = max(left_n, right_n)  # 1.0 if any support leg touches
    angle_bonus = 1.0 / (1.0 + abs(angle_n))
    contact_stability = (
        CONTACT_WEIGHT * proximity * contact_flag * speed_bonus * angle_bonus
    )

    # ---------- 5) inferred success bonus ----------
    is_success = (
        R_next < SUCCESS_DIST_THRESHOLD and
        speed < SUCCESS_SPEED_THRESHOLD and
        abs(angle_n) < SUCCESS_ANGLE_THRESHOLD and
        contact_flag > 0.5
    )
    success_bonus = SUCCESS_BONUS if is_success else 0.0

    # ---------- aggregate ----------
    total_reward = (
        progress_reward + soft_landing + angle_penalty +
        contact_stability + success_bonus
    )

    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "angle_penalty": angle_penalty,
        "contact_stability": contact_stability,
        "success_bonus": success_bonus,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 接触信号（支撑腿）完全未使用，导致 agent 学会 hover 但不会着陆；angular_velocity_penalty 为僵尸组件。
- **behavior**: agent 成功飞到目标附近并减速，存活 956 步，但无法完成最终的着陆动作，19/20 超时。
- **signal**: 缺少“接触目标垫”的正向激励，以及“成功着陆”的一次性强信号。
- **level**: Level 2
- **hypothesis**: 新增接触稳定性奖励和推断成功 bonus 将推动 agent 完成 touchdown，填补最后的 58 分缺口，同时保留已习得的减速和接近行为。
- **risk**: 推断成功条件可能在某些非目标垫的软接触时刻产生假阳性奖励，但严格的四阈值会大幅降低概率；若仍出现 exploit，后续可收紧阈值或加入排除非目标区域的门控。
