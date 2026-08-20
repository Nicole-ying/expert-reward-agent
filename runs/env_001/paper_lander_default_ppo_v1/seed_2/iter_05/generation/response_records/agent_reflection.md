# Response Record

`evidence`：当前 final-policy 得分 73.07，episode 平均长度 414 步，terminated=15/20（成功着陆），但仍有 4/20 早期崩溃（<150 步且 score<-50）。landing_approach 占 magnitude_share 93.2%，active_rate 98.6%，形成稠密 proxy；distance_progress 路径无关总额仅 ~2.18（4.4%）。上一轮将稀疏 binary proxy 改为稠密 geometric mean 后，episode 长度从 68 跃升至 414 且得分首次转正，但未引入效率激励。

`behavior_diagnosis`：agent 已学会接近并着陆（75% terminated），但极其缓慢（平均 414 步）。landing_approach 稠密 proxy 奖励"靠近+低速+小角度"的任意状态，不区分快速抵达与缓慢徘徊，形成中等分平台。4 个早期崩溃 episode 表明着陆稳健性仍有不足。

`signal_completeness`：progress（distance_progress）正确但贡献微弱；稳定性约束（stability_penalty）合理；着陆引导（landing_approach）稠密但缺失时间维度。任务明确要求"尽可能快地到达"，当前奖励完全缺失效率/时间压力信号，属于必要职责缺失。

`selected_level`：Level 2 — 触发条件为"必要信号缺失"（时间效率激励）。landing_approach 形成稠密 proxy 中等分平台，匹配 `dense_to_task_event` 证据模式，但本轮的精确干预方向是增加时间惩罚以直接填补效率信号的空白，而非立即将 proxy 改造为离散事件。

`selected_intervention`：新增组件 `time_penalty`，固定每步 -0.02，作为唯一修改目标。保持 distance_progress、stability_penalty、landing_approach 完全不变。

`falsifiable_hypothesis`：固定步数惩罚使慢速 episode 累计更多负奖励，agent 将学习缩短从出发到着陆的时间，从而提升外部 score；-0.02 尺度温和（占当前 landing_approach 均值 ~16%），应推动效率改善而不引发为逃避惩罚而主动崩溃。

`expected_next_round`：episode 平均长度应从 ~414 下降，同时 terminated 比例保持或提升（不应显著恶化）；landing_approach 的 magnitude_share 可能略微下降但仍是主信号；外部 score 应上升；早期崩溃 episode 数量不应增加。

`main_risk`：时间压力可能诱使 agent 在探索阶段选择更快但更危险的轨迹，导致崩溃率上升；若崩溃增加的负分超过效率提升的正分，score 可能停滞甚至下降。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next observation variables
    prev_x, prev_y = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress signal: reward getting closer to target, penalize moving away
    prev_dist = (prev_x**2 + prev_y**2) ** 0.5
    curr_dist = (x**2 + y**2) ** 0.5
    reward_dist = 2.0 * (prev_dist - curr_dist)

    # 2. Stability constraint: light penalty on high speeds and large angles
    reward_stability = (
        -0.008 * abs(vx) -
        0.008 * abs(vy) -
        0.008 * abs(angle) -
        0.008 * abs(omega)
    )

    # 3. Dense landing approach guidance: geometric mean of bounded factors
    prox = max(0.0, 1.0 - curr_dist / 2.0)
    vel_mag = abs(vx) + abs(vy)
    vel_factor = max(0.0, 1.0 - vel_mag / 2.0)
    ang_factor = max(0.0, 1.0 - abs(angle) / 0.5)

    # Geometric mean prevents product collapse while preserving joint requirement
    approach_quality = (prox * vel_factor * ang_factor) ** (1.0 / 3.0)

    # Contact factor: partial credit without legs, full credit with both
    both_legs = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_factor = 0.3 + 0.7 * both_legs

    reward_landing = 0.3 * approach_quality * contact_factor

    # 4. Time penalty: small fixed per-step cost to incentivize faster completion
    time_penalty = -0.02

    total_reward = reward_dist + reward_stability + reward_landing + time_penalty

    components = {
        "distance_progress": reward_dist,
        "stability_penalty": reward_stability,
        "landing_approach": reward_landing,
        "time_penalty": time_penalty
    }

    return float(total_reward), components
```
