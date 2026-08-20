# Response Record

## 诊断与决策

**evidence**：当前代码（iter 5-6）得分 -236.15，landing_reward 的 active_rate=0%（从未触发），shaping_reward 活跃且均值为 894.24，但被 exploit——agent 持续收集近原点低姿态奖励而不着陆，15/20 回合被截断。

**behavior_diagnosis**：agent 学会了在目标附近悬停/徘徊来持续收割 shaping 奖励，从未完成双脚着陆，episode 长（835 步）但无任务进展。

**signal_completeness**：当前 reward 缺少可达的着陆完成信号——landing_reward 仅在双脚同时接触时触发（稀疏二值门控），且 shaping 提供的是可被无限收割的持续邻近奖励，缺少从"靠近"到"着陆"的中间梯度。

**selected_level**：Level 2。当前骨架已连续两轮远差于 best（-236 vs 149），需回退到 best 骨架（approach_reward + stability_penalty + soft_landing_proxy）并做一次有证据的结构修改；直接复用 best 无修改不满足"必须做新的、有证据的修改"规则。

**selected_intervention**：修改 `soft_landing_proxy` 中的接触信号，从二值乘积 `next_obs[6] * next_obs[7]`（仅双脚同时触地=1，否则=0）改为连续平均 `(next_obs[6] + next_obs[7]) / 2.0`（无接触=0，单脚=0.5，双脚=1.0），其余组件不变。这属于 `product_to_noncollapsing_joint` 变换。

**falsifiable_hypothesis**：连续接触信号为部分着陆（单脚先触地）提供梯度，使 soft_landing_proxy 不再仅在双脚触地瞬间才产生非零反馈，应提升着陆成功率并降低徘徊截断率。

**expected_next_round**：soft_landing_proxy 的 active_rate 应显著上升（不再仅在双脚同时触地时触发），terminated 比例上升，truncated 比例下降，episode_length 缩短，score 应改善并可能刷新 best。

**main_risk**：单脚触地获得 50% 奖励可能鼓励不稳定着陆姿态，如果 agent 停在单脚状态不再优化到双脚，需要后续调整 stability_penalty 权重或引入姿态约束。

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

    # 3. Soft landing proxy: continuous contact signal replaces binary product.
    # (left + right)/2 gives 0/0.5/1.0 gradient for partial landings.
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

    total_reward = approach_reward + stability_penalty + soft_landing_proxy

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```
