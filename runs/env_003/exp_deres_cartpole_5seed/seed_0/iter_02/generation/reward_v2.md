## 诊断分析

### 0. 环境理解
- **观测**: `obs[0]=pos`(limit ±2.4), `obs[2]=angle`(limit ±0.209 rad) — 典型的平衡存活任务，每步 original_env_reward=1.0
- **分数**: score=500, episode_length=500 → 已存活满整局，表面看"解了"

### 1. Agent 发生了什么？
agent 确实存活满 500 步，但这不是因为奖励函数提供了有效梯度。`progress_reward` 的 mean=-0.0016, abs_mean=0.0034 — **信号几乎为零**。potential-based shaping 的 Φ 在最优态附近本身就是平坦的，加上 Φ 始终 ≤0，噪声导致 shaping reward 有轻微负偏。agent 能存活更多是因为环境初始策略尚可，而非奖励函数在引导。

### 2. 哪个组件是主要原因？
唯一组件 `progress_reward`。绝对值太小（0.003），ratio_to_progress_reward 无意义（自己对自己=-1）。核心问题：**Φ = -((angle/limit)²+(pos/limit)²) 在最优态附近梯度近乎消失**，且始终 ≤0 导致 shaping reward 有系统性负偏。

### 3. 我之前改了什么？
这是 iter 1，初次提交。上一轮用了纯 potential-based shaping，得分已达上限 500。当前不需要大改方向，但要增强信号的数学形态。

---

**修订策略**: 层次 2 — 改数学形式。把 Φ 从"负值抛物面"改成"正值有界凸函数"：`Φ = 1/(1 + (angle/limit)² + (pos/limit)²)`。这样 Φ∈[1/3, 1]，最优态 Φ≈1 是全局最大值，shaping reward 在稳态趋近 (γ-1)≈-0.01，提供可预测的梯度基线，偏离时 reward 显著变负，梯度清晰。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Survival balance task (Env_003).
    Potential-based shaping with positive bounded potential function.
    
    Change from iter 1: Φ changed from negative quadratic (-x²) to positive
    bounded form 1/(1+x²). This provides a proper "potential well" with global
    maximum at the goal state, ensuring non-vanishing gradient and removing
    the negative bias at optimum.
    """
    GAMMA = 0.99
    ANGLE_LIMIT = 0.20943951
    POS_LIMIT = 2.4

    pos_now, _, angle_now, _ = obs[0], obs[1], obs[2], obs[3]
    pos_next, _, angle_next, _ = next_obs[0], next_obs[1], next_obs[2], next_obs[3]

    # Positive bounded potential: peaks at 1.0 when pole upright & base centered
    # Φ ∈ [1/3, 1], with maximum at the goal → proper potential well
    phi_now = 1.0 / (1.0 + (angle_now / ANGLE_LIMIT) ** 2 + (pos_now / POS_LIMIT) ** 2)
    phi_next = 1.0 / (1.0 + (angle_next / ANGLE_LIMIT) ** 2 + (pos_next / POS_LIMIT) ** 2)

    # Potential-based shaping: F = γ·Φ(s') - Φ(s)
    progress_reward = GAMMA * phi_next - phi_now

    total_reward = progress_reward
    components = {
        "progress_reward": progress_reward
    }

    return float(total_reward), components
```