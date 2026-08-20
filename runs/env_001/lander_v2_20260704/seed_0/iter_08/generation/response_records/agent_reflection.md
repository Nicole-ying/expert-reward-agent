# Response Record

## 诊断与干预

`evidence`：当前骨架首次刷新best至167.99，但proximity_reward以持续正值占magnitude_share的88.1%（episode_sum_mean=491.13），而实际任务得分仅167.99，agent在647步中缓慢着陆（terminated=17/20），存在"近目标徘徊、延迟着陆"的hover exploit迹象。

`behavior_diagnosis`：agent学会了停在目标附近持续收取proximity_reward（每步约0.76），仅偶尔触地（contact active_rate仅14.9%），最终虽能着陆但效率低下——这正是持续正值状态奖励抑制终止的典型模式。

`signal_completeness`：当前奖励缺少"到达后应尽快完成"的激励。proximity_reward作为持续状态值（`1/(1+d)`始终>0），使"待在好状态"比"完成任务并终止"更有利可图，这是数学形态问题而非尺度问题。

`selected_level`：Level 2。证据直接命中`state_to_improvement`的适用条件——"占据好状态即可持续获奖"，且这不是单纯系数过强而是数学形态固有缺陷（任何正系数都会产生反终止激励），Level 1降系数无法根除。

`selected_intervention`：将`proximity_reward`从持续状态值变换为状态改善量（delta），即`80 * (proximity(next_obs) - proximity(obs))`，系数80使整回合总贡献与contact_reward（~51）可比。

`falsifiable_hypothesis`：改为delta后，proximity不再奖励"停留"，只奖励"靠近"；一旦到达目标附近，proximity delta趋近零，contact_reward将成为主导正信号，推动agent尽快完成双腿着陆。episode_length应显著缩短，contact active_rate应上升。

`expected_next_round`：proximity_reward的episode_sum_mean从~491降至~40-60量级，magnitude_share不再支配；contact_reward的magnitude_share上升；episode_length从~647明显下降；score因着陆效率提升而改善。

`main_risk`：delta形态使proximity可能为负（远离目标时），若初始探索阶段agent随机远离目标，负proximity累积可能加剧早期失败；但结合velocity_penalty和angle_penalty的约束，以及接近目标后contact_reward的吸引，整体应可收敛。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步与下一步的相对目标距离
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 非线性接近度：有界且近目标时梯度更强
    prox_current = 1.0 / (1.0 + dist_current)
    prox_next = 1.0 / (1.0 + dist_next)

    # 状态改善量：奖励靠近，惩罚远离，停留则趋零（系数使整回合总量与contact可比）
    proximity_improvement = 80.0 * (prox_next - prox_current)

    # 稳定约束：抑制高速
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    velocity_penalty = -0.1 * (abs(x_vel) + abs(y_vel))

    # 姿态约束：抑制大幅倾斜
    angle_penalty = -0.05 * abs(next_obs[4])

    # 着陆信号：奖励双腿触地
    contact_reward = 0.3 * (next_obs[6] + next_obs[7])

    total_reward = proximity_improvement + velocity_penalty + angle_penalty + contact_reward

    components = {
        'proximity_improvement': proximity_improvement,
        'velocity_penalty': velocity_penalty,
        'angle_penalty': angle_penalty,
        'contact_reward': contact_reward
    }

    return float(total_reward), components
```
