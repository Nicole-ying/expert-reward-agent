`evidence`：本轮soft_landing_proxy以97.8%的signed_share支配奖励，episode_sum_mean=82.79，而approach_reward仅1.12（结构上每episode最多累加初始距离约1个单位），stability_penalty=-0.54几乎可忽略；20/20 episode成功terminate，score范围[29.79, 259.37]说明着陆质量差异巨大；上一轮（iter 7）刚刷新best到198.14，是该骨架家族的第二次有效迭代。

`behavior_diagnosis`：策略已学会稳定着陆（全部terminate），但approach_reward受限于差分形式天生只能累计约1/episode，无法对快速接近形成有效激励；软着陆proxy完全主导学习信号，导致策略可能以任意速度接近目标，只要最终着陆姿态良好即可获高分——这解释了score的上限和episode_length=406偏长的现象。

`signal_completeness`：当前奖励具备接近引导（approach_reward）、姿态稳定约束（stability_penalty）和着陆质量评估（soft_landing_proxy）三个职责，但缺少任务明确要求的"尽量少使用引擎推力"的节能信号。engine_penalty完全缺失，属于必要职责不完备。

`selected_level`：Level 2。触发条件：必要职责（引擎节能）缺失，且该职责无法通过调整现有组件系数（Level 1）补全——approach_reward的结构性上限无法通过系数修复，stability_penalty已合理，soft_landing_proxy的系数调整不会引入节能激励。

`selected_intervention`：新增`engine_penalty`组件——对任何非零action（即使用引擎）施加小幅惩罚，鼓励策略仅在必要时点火。

`falsifiable_hypothesis`：加入轻量引擎惩罚后，策略应在保持成功着陆的前提下减少不必要的引擎使用，从而缩短episode长度（更高效接近）且着陆质量不降；若惩罚过重导致策略不敢用主引擎减速而着陆失败，则预期truncated比例上升或score显著下降。

`expected_next_round`：engine_penalty的active_rate应接近非零action的实际使用率（预计30-70%），episode_sum_mean为小额负值（若引擎使用减半则约-1.0）；episode_length应下降（更快接近）；soft_landing_proxy保持或略有下降；score可能因新增惩罚项而小幅下降，但外部着陆行为应更高效。

`main_risk`：引擎惩罚可能使策略在需要强力制动时犹豫（如接近平台时不敢开主引擎），导致硬着陆、crash或terminate条件从success变为failure，表现为terminated比例下降或score分布恶化。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. Main learning signal: progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = d_prev - d_next

    # 2. Stability constraint: light penalty on large speeds, tilt, and angular velocity.
    speed = abs(next_obs[2]) + abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    w_speed = 0.001
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = -w_speed * speed - w_angle * angle - w_angvel * ang_vel

    # 3. Soft landing proxy: continuous contact signal.
    contact_avg = (next_obs[6] + next_obs[7]) / 2.0
    dist = d_next
    k_dist = 2.0
    k_speed = 1.0
    k_angle = 5.0
    w_proxy = 5.0
    soft_landing_proxy = (w_proxy * contact_avg *
                          (2.718281828 ** (-k_dist * dist)) *
                          (2.718281828 ** (-k_speed * speed)) *
                          (2.718281828 ** (-k_angle * angle)))

    # 4. Engine efficiency penalty: discourage unnecessary engine use.
    engine_used = 0.0 if action == 0 else 1.0
    w_engine = 0.005
    engine_penalty = -w_engine * engine_used

    total_reward = approach_reward + stability_penalty + soft_landing_proxy + engine_penalty

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "engine_penalty": engine_penalty
    }

    return float(total_reward), components
```