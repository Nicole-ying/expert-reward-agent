# Response Record

# 设计理由
将 `contact_bonus` 从“转移事件奖励”改为“连续接触‑距离奖励”。上一轮的转移事件奖励平均值高达 66.4，推测 agent 通过在平台附近反复弹跳获得多次 bonus，导致多数 episode 被截断且稳定停靠很少。新形式为 `next_lc * next_rc * 1/(1+5*dist)`，只在双接触且靠近着陆中心时持续给出正奖励，消除重复领取的漏洞，同时保留原有的 proximity/velocity/orientation 惩罚以维持基础约束。系数 1.0 在中心双接触时每步给出约 1 的正信号，量级上仍能引导稳定停靠。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    av = next_obs[5]
    next_lc = next_obs[6]
    next_rc = next_obs[7]

    dist = (x**2 + y**2) ** 0.5

    proximity_reward = -0.1 * (x**2 + y**2)

    vel_gate = 1.0 / (1.0 + 5.0 * dist)
    velocity_penalty = -0.1 * (vx**2 + vy**2) * vel_gate

    orientation_penalty = -0.5 * (angle**2) - 0.1 * (av**2)

    contact_proximity = next_lc * next_rc * (1.0 / (1.0 + 5.0 * dist))

    total_reward = proximity_reward + velocity_penalty + orientation_penalty + contact_proximity

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_proximity': contact_proximity
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: `contact_bonus` 平均 66.4，`terminated=4/20`，`len=842.75`，`score_range` 包含 302 但多数 episode 被截断。  
- **behavior**: agent 在少数 episode 中成功着陆，多数 episode 可能在目标平台附近弹跳，通过反复触发双接触事件获取高额奖励，但未能稳定停靠。  
- **signal**: 接触奖励作为稀疏转移事件，鼓励了非稳定行为；缺少对“双接触且静止在中心”的持续性激励。  
- **level**: Level 2（数学结构变换：稀疏事件→连续 proximity-gated 奖励）。  
- **hypothesis**: 用距离衰减的连续接触奖励替代转移事件，使 agent 只有在保持双接触且靠近中心时才能持续获得正回报，从而学会稳定停靠。  
- **risk**: 若 proximity_reward 的负向推动不足，agent 可能在未接触时徘徊；但 -0.1*dist² 仍能提供足够的基础引力。
