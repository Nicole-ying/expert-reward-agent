## 分析

1. **evidence**：本轮 score=-7.39，20/20 truncated（均跑满 1000 步未终止），soft_landing_proxy 的 active_rate=0.0% 完全哑火，engine_penalty 以 -4.89 主导负向，approach_reward=+1.32 说明 agent 确实在靠近平台但从未完成着陆。

2. **behavior_diagnosis**：agent 学会了向平台靠近（approach_reward 为正），但因 soft_landing_proxy 的乘积形态在 contact_avg=0 时整体坍缩为零，着陆阶段完全无梯度信号，策略无法学会最终着陆动作，只能徘徊至 truncation。

3. **signal_completeness**：progress 信号（approach_reward）和稳定约束（stability_penalty）职责正常；但着陆完成信号（soft_landing_proxy）因乘积坍缩导致完全不可达——contact_avg 作为乘性闸门，未触地时整个组件归零，丧失了着陆配置的中间梯度引导。

4. **selected_level**：Level 2，触发条件为 `product_to_noncollapsing_joint`——soft_landing_proxy 是典型的多因子乘积坍缩，active_rate=0% 直接证实即使 agent 已靠近平台，着陆信号也从未激活。

5. **selected_intervention**：将 soft_landing_proxy 从单一乘积结构拆分为两个加法组件——`proximity_quality`（始终活跃的着陆姿态质量信号，用有界有理函数替代指数乘积）和 `contact_bonus`（触地加法奖励），同时移除本轮引入的 engine_penalty（以 best 代码为基底）。

6. **falsifiable_hypothesis**：proximity_quality 提供始终非零的"靠近+低速+正姿"梯度，contact_bonus 提供触地额外激励，二者之和使 agent 能先学会接近平台再学会着陆，而非在无梯度区域卡死；移除 engine_penalty 消除其对最终着陆机动的抑制。

7. **expected_next_round**：proximity_quality 和 contact_bonus 的 active_rate 应显著高于本轮 soft_landing_proxy 的 0%，episode_length 应从 1000 下降（有自然终止），score 应大幅改善并接近或超过 best 的 198。

8. **main_risk**：proximity_quality 始终活跃可能产生"近平台徘徊"的局部最优——agent 在平台附近悬停收集 proximity_quality 而不着陆；若出现此情况，下一轮需将 proximity_quality 从状态值转为状态改善量（state_to_improvement）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. Progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = d_prev - d_next

    # 2. Stability constraints: light penalty on speed, tilt, and angular velocity.
    speed = abs(next_obs[2]) + abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * speed - 0.01 * angle - 0.005 * ang_vel

    # 3. Proximity quality: always-active bounded signal for landing-ready configuration.
    #    Rational functions 1/(1+k*x) never collapse to exactly zero,
    #    providing gradient toward low distance, low speed, and upright attitude.
    dist = d_next
    proximity_quality = 2.0 * (1.0 / (1.0 + 2.0 * dist)) * (1.0 / (1.0 + speed)) * (1.0 / (1.0 + 5.0 * angle))

    # 4. Contact bonus: additive reward for foot contact, decoupled from proximity product.
    contact_avg = (next_obs[6] + next_obs[7]) / 2.0
    contact_bonus = 2.0 * contact_avg

    total_reward = approach_reward + stability_penalty + proximity_quality + contact_bonus

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "proximity_quality": proximity_quality,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```