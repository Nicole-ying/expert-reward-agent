# 设计理由
agent 在 1000 步中从未触发双接触（contact_reward=0），处于徘徊状态且所有 episode 均被截断。我将 contact 组件从密集状态奖励（每步双接触给固定 +1）改为转移事件奖励（仅从“未双接触”变为“双接触”的那一步给予一次 +1），以提供稀疏但明确的目标信号，避免密集奖励因从未触发而缺失梯度。其他组件系数保持不变。

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

    # 转移事件奖励：仅在从未双接触变为双接触的步给予一次奖励
    prev_contact = obs[6] * obs[7]
    curr_contact = next_lc * next_rc
    contact_bonus = 1.0 * curr_contact * (1.0 - prev_contact)

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
- **evidence**: terminated=0/20, truncated=20/20, contact_reward 平均 0.0（从未双接触）
- **behavior**: agent 徘徊 1000 步，始终没有同时双腿着陆，无法触发任何终止条件
- **signal**: 密集的双接触奖励因事件零触发而完全无效，缺乏引导着陆的梯度
- **level**: Level 2（数学结构变换：持续状态奖励 → 转移事件奖励），因为 active_rate≈0 且密集形式无法提供学习信号
- **hypothesis**: 转移事件奖励在首次实现双接触时提供明确的正脉冲，将引导策略同时接触双腿，从而恢复着陆行为
- **risk**: 若 proximity 惩罚与速度惩罚的尺度仍然主导，agent 在接近目标时可能因害怕惩罚而不愿探索接触；但历史上 iter 2 已证明该组合有效