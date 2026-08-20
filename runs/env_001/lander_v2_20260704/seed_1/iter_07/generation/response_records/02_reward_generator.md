# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # unbox next_obs for the state after the action
    next_x  = next_obs[0]
    next_y  = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle     = next_obs[4]
    next_angvel    = next_obs[5]
    left_contact   = next_obs[6]
    right_contact  = next_obs[7]

    # --- helper for exponential (e^z) ---
    def exp(z):
        return 2.718281828 ** z

    # --- helper for tanh using e^(-2z) ---
    def tanh(z):
        e_neg = exp(-2.0 * z)
        return (2.0 / (1.0 + e_neg)) - 1.0

    # ========== 1. approach_signal (main driving term) ==========
    # encourage being close to the landing pad: r ∈ [0, 2]
    dist_next = (next_x**2 + next_y**2) ** 0.5
    alpha = 2.0
    r_approach = 2.0 * (1.0 - tanh(alpha * dist_next))

    # ========== 2. stability_penalty (safety / smoothness) ==========
    w_vel    = 0.02
    w_angle  = 0.02
    w_angvel = 0.02
    r_stability = (
        - w_vel    * (abs(next_vx) + abs(next_vy))
        - w_angle  * abs(next_angle)
        - w_angvel * abs(next_angvel)
    )

    # ========== 3. landing_bonus (soft proxy for successful landing) ==========
    # only fires when the agent is almost perfectly on the pad,
    # very slow, upright, and both legs touching.
    contact_factor = 0.5 * (left_contact + right_contact)
    angle_factor   = 1.0 / (1.0 + 2.0 * next_angle**2)

    w_land = 30.0
    r_land = w_land * (
        exp(-10.0 * next_x**2) *
        exp(-10.0 * next_y**2) *
        exp(-5.0  * next_vx**2) *
        exp(-5.0  * next_vy**2)
    ) * angle_factor * contact_factor

    # ========== total ==========
    total_reward = r_approach + r_stability + r_land

    components = {
        'approach_signal':  r_approach,
        'stability_penalty': r_stability,
        'landing_bonus':     r_land,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及其角色

- **approach_signal (主学习信号)**：`2 * (1 - tanh(2.0 * dist))`  
  这是一个**稠密的正向接近信号**，当飞行器距离目标平台越近时奖励越高（最大值 2，远距离接近 0）。  
  使用 `tanh` 实现自然的饱和特性，避免远距离时出现过大的梯度差异，也避免 agent 在极近距离时被过强的梯度推离稳定状态。  
  **角色**：驱动 agent 向着中央平台移动，是奖励的核心驱动力。

- **stability_penalty (稳定/安全约束)**：对水平/垂直线速度、机体倾角和角速度施加**轻量线性惩罚**。  
  权重很小（各 0.02），作用类似“方向盘”而非“刹车”，只在高速度或大角度时提供纠正信号，不致于让 agent 害怕移动。  
  **角色**：促使着陆过程平缓，为最终的稳定着陆创造必要条件。

- **landing_bonus (任务完成近似信号，proxy)**：基于多条件连续乘积的**软着陆奖赏**。  
  利用 `exp` 函数将水平位置、高度、线速度、腿接触和倾角组合成一个仅在“几乎完美着陆”时显著激活的
