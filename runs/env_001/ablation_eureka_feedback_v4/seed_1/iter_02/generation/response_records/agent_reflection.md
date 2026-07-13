# Response Record

# 设计理由
将 contact_bonus 从持续每步奖励双腿接触改为仅奖励接触转移事件（从未接触到同时接触的瞬间），解决 agent 通过反复蹭接触刷分但延迟稳定着陆的问题。使用 `max(0.0, next_lc * next_rc - prev_lc * prev_rc)` 捕捉首次双腿同时接触的瞬间，系数保持 10.0，但奖励从多步累积变为一次性事件。其他组件保持不变，确保接近、减速和姿态约束仍持续提供引导。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    prev_lc = obs[6]
    prev_rc = obs[7]

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

    contact_transition = max(0.0, (next_lc * next_rc) - (prev_lc * prev_rc))
    contact_bonus = 10.0 * contact_transition

    total_reward = proximity_reward + velocity_penalty + orientation_penalty + contact_bonus

    components = {
        'proximity_reward': proximity_reward,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'contact_bonus': contact_bonus
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: contact_bonus 均值 147.5 远超其他组件（proximity -15.99），terminated=9/20 表明持续接触常未触发终止。
- **behavior**: agent 靠在平台上持续获取接触奖励刷分，但未稳定停靠导致截断或缓慢徘徊。
- **signal**: 缺少对一次性成功着陆的激励，现有持续接触奖励助长了不终止的刷分行为。
- **level**: Level 2 — 将持续状态值改为转移事件，数学形态从每步连续奖励变成一次性事件奖励。
- **hypothesis**: 只奖励接触瞬间能促使 agent 在着陆后快速满足终止条件，减少无谓徘徊。
- **risk**: 过早接触但尚未稳定时可能丧失后续奖励动机，但 orientation 和 velocity 惩罚仍会推动姿态稳定。
