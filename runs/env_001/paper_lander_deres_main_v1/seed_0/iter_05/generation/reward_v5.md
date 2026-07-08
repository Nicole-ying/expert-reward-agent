Now let me do the diagnosis.

## 诊断

**0. 环境理解：** 2D 飞行器着陆任务。飞行器从顶部出发，目标是平稳降落在中央平台上（位置接近0、速度接近0、姿态直立、双腿触地）。`original_env_reward` 的 ratio 为正（+16.6），与 progress 同向，不存在 misalignment 信号。

**1. Agent 发生了什么？** 
- Iter 3（最佳）：score=146.36，episode_length=1000（达到上限），approach_quality 均值 0.094，progress 均值 0.002。agent **在目标附近徘徊/悬停**——它学会了"靠近 + 慢速 + 直立"来稳定收割 approach_quality，但没有足够的动力完成最后的着陆动作（双腿触地 + 静止）。episode 达到最大长度表明它不 crash，但也不完成任务。
- Iter 4：新增 contact_bonus 后得分反而从 146 降到 128（下降 -12%），contact_bonus episode_sum=29.15 占 total 的 37%。结合知识库 `contact_reward_hacking` 模式：contact 奖励高但 external score 低 → contact 奖励被 exploit（可能反复弹跳触地刷奖励）。

**2. 哪个组件是主因？**
- Iter 4 的 contact_bonus 是得分下降的直接原因（唯一改动 → 得分下降）。它的 ratio_to_progress=22，nonzero_rate=65%，说明条件太宽松（只要 prox 门控 + 任意腿接触就奖励），agent 利用它刷分而非真正着陆。
- Iter 3 的根本问题是 approach_quality 与 progress 的比例 37:1 严重失衡，且缺乏"完成任务"的终端引导，导致徘徊。

**3. 我之前改了什么？**
- Iter 3→4：添加了 `contact_bonus = prox_factor * (left_contact + right_contact) * 0.05`。得分下降。本轮不应再尝试"加简单的 contact 奖励"。

**修改决策：** 根据 revert 规则，回到 iter 3 代码，在此基础上做一个新修改。本次的修改方向：**添加一个严格门控的 landing proxy**——它只在接近目标 + 低速 + 直立 + **双腿同时触地**时触发，直接奖励"完成着陆"这个终端状态。这与 iter 4 的 contact_bonus 不同：(1) 要求双腿同时触地而非任意腿；(2) 要求低速和直立门控；(3) 作为独立 landing 信号而非弱 contact 奖励。

这对应知识库 `contact_reward_hacking` 的修复建议："contact 奖励只能在 near target + low speed + stable angle + both supports 下触发"。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next states
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]

    # 1. Core learning signal: progress towards target (0,0)
    dist_current = (x ** 2 + y ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_current - dist_next  # positive when approaching

    # 2. Continuous approach quality shaping
    #    Three bounded factors [0,1], product ensures all must be satisfied
    prox_factor = 1.0 / (1.0 + 5.0 * dist_next)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_angle))
    approach_quality = prox_factor * speed_factor * angle_factor  # [0, 1], dense

    # 3. Orientation penalty (gentle, keeps the craft upright)
    attitude_penalty = -0.01 * abs(next_angle)

    # 4. Landing proxy: rewards the terminal landing state.
    #    Only activates when ALL conditions are met:
    #    near target + low speed + upright + BOTH legs contacting.
    #    Shares the same quality factors as approach_quality to ensure
    #    the agent only gets this bonus during a genuine controlled landing.
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_legs = left_contact * right_contact  # 1.0 only when both touch
    landing_proxy = approach_quality * both_legs * 0.3  # max 0.3 per step

    w_approach = 0.15  # unchanged from iter 3
    approach_quality_reward = w_approach * approach_quality
    total_reward = progress + approach_quality_reward + attitude_penalty + landing_proxy

    components = {
        "progress_reward": progress,
        "approach_quality_reward": approach_quality_reward,
        "attitude_penalty": attitude_penalty,
        "landing_proxy": landing_proxy,
    }

    return float(total_reward), components
```