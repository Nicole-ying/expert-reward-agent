## 诊断

### 1. Agent 发生了什么？
- **Episode 长度 68.4（<150），100% early terminal → 全部 crash。**
- Score=-115，per-step total=-1.7，distance_reward mean=-0.97 → agent 在向目标靠近但无法减速/着陆。
- soft_landing_continuous 的 **nonzero_rate 仅 0.55%**（≈每 180 步触发一次，但 episode 才 68 步）→ 信号基本死透。

### 2. 哪个组件是主因？
- `soft_landing_continuous`：4 因子乘积（proximity × speed_factor × angle_factor × contact_bonus），其中 `contact_bonus` 在飞行中恒为 0，把整个信号乘没了。连续形式没解决问题，反而比稀疏版更差（mean 从 0.028 降到 0.013）。
- `stability_penalty` abs_contrib 仅 0.47%，太弱无法引导减速。
- `distance_reward` 只告诉 agent "靠近原点"，不告诉它"减速/稳定/触地"。

### 3. 我之前改了什么？
- Iter 1→2：把 soft_landing 从稀疏二值改成连续乘积 → 得分反而从 -111.87 降到 -115.29，信号更弱了。
- **同一骨架家族已迭代 2 轮，最佳得分 -111.87，远未达到合理着陆分数（>0），触发层次 3：换骨架。**

### 修改方向
换用 **potential-based shaping**（势能塑形），将 dist + speed + angle 统一纳入势能函数，天然提供"靠近 + 减速 + 稳定"的稠密梯度。再加独立的 contact bonus（加法而非乘法），不再被乘积归零。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 势能塑形 (Potential-Based Shaping) ===
    # 原理：F = γ * Φ(s') - Φ(s)，是唯一保证最优策略不变的塑形方式。
    # Φ(s) = -(dist + α*speed + β*|angle|)，同时引导靠近、减速、姿态稳定。
    # 相比 distance_reward = -dist'（只关心绝对位置），势能塑形奖励"改善量"，
    # 提供更稠密的梯度：靠近有奖、减速有奖、摆正有奖。

    # 当前状态
    dist = ((obs[0] - 0.0) ** 2 + (obs[1] - 0.0) ** 2) ** 0.5
    speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    angle = abs(obs[4])

    # 下一状态
    dist_next = ((next_obs[0] - 0.0) ** 2 + (next_obs[1] - 0.0) ** 2) ** 0.5
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_next = abs(next_obs[4])

    # 势能函数：负的加权状态代价
    alpha_speed = 0.5   # 速度权重，使 speed 分量与 dist 分量量级可比
    beta_angle = 0.5    # 角度权重，使 angle 分量与 dist 分量量级可比
    phi_now = -(dist + alpha_speed * speed + beta_angle * angle)
    phi_next = -(dist_next + alpha_speed * speed_next + beta_angle * angle_next)

    gamma = 0.99  # 接近 1，轻微折扣提供时间偏好
    potential_shaping = gamma * phi_next - phi_now

    # === 独立接触奖励（加法，不被乘积归零） ===
    # 左右腿接触均为 [0,1] 连续值
    contact_bonus = (next_obs[6] + next_obs[7]) * 0.1

    # === 总奖励 ===
    total_reward = potential_shaping + contact_bonus

    components = {
        'potential_shaping': potential_shaping,
        'contact_bonus': contact_bonus,
        'total_reward': total_reward
    }

    return float(total_reward), components
```