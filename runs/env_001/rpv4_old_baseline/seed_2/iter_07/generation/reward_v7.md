**反思分析**

1. **根因判断**：score=15.37远低于target=200，且landing_bonus占据了89.6%的magnitude_share，但其active_rate仅3.7%，属于极其稀疏的信号。其余shaping组件（descent_shaping、horiz_penalty、vx_penalty等）量级过小，无法提供有效的学习梯度。整个奖励函数结构错误，必须重写主信号框架。

2. **组件逐一分析**：
   - **landing_bonus**：magnitude_share极高（89.6%），active_rate仅3.7%，奖励稀疏且量级过大，导致agent被单一稀疏信号驱动，学习严重受阻。必须移除这种一次性巨大奖励，改为密集、渐进的信号。
   - **descent_shaping**：活跃率100%但magnitude_share仅1.2%，惩罚力度太弱，无法引导下降行为。
   - **horiz_penalty, vx_penalty, orient_penalty, angvel_penalty**：均活跃但量级极低（合计magnitude_share约1%），几乎不起作用。
   - **contact_reward**：活跃率25.8%，量级8.1%，有一定作用但远不足以弥补稀疏的landing_bonus。
   - **time_penalty**：量级0.1%，可忽略。

3. **修改策略**：
   - 用**指数状态势能奖励**（state_exponential_reward）作为核心密集信号，每一步根据当前状态与理想着陆状态的“距离”给出连续奖励，使agent在任何位置都能获得梯度指导。
   - 保留**接触奖励**鼓励腿部触地，但调低幅度，避免量级过大。
   - 加入**下降进度奖励**（descent_bonus），直接奖励高度降低，加速agent向下移动的学习。
   - 保留轻微**时间惩罚**，鼓励快速完成任务。
   - 完全移除原landing_bonus和所有无效的shaping惩罚项，避免信号淹没。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next state
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- Core dense signal: exponential state goodness ----
    # Encourage all state components to approach zero (landed, upright, still)
    # Higher squared penalties for y, angle, and velocities to drive descent and stability
    squared_error = (0.2 * x**2 + 1.0 * y**2 + 0.5 * vx**2 + 0.5 * vy**2 +
                     5.0 * body_angle**2 + 2.0 * angvel**2)
    state_goodness = 10.0 * (2.71828 ** (-squared_error))  # use e^(-error)
    # Maximum ~10 when fully landed, decays gracefully with any deviation

    # ---- Contact reward: encourage touching the platform ----
    contact_reward = (left_contact + right_contact) * 0.5

    # ---- Descent bonus: reward downwards progress ----
    # y decreases when moving down, so obs[1]-next_obs[1] is positive on descent
    descent_bonus = 1.0 * max(obs[1] - next_obs[1], 0.0)

    # ---- Small per‑step penalty to discourage lingering ----
    time_penalty = -0.02

    total = state_goodness + contact_reward + descent_bonus + time_penalty

    components = {
        "state_goodness": state_goodness,
        "contact_reward": contact_reward,
        "descent_bonus": descent_bonus,
        "time_penalty": time_penalty
    }
    return float(total), components
```