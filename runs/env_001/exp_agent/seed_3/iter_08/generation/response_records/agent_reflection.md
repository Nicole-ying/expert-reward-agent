# Response Record

## 诊断

### 1. Agent 发生了什么？
**Crash。** Episode length 70.5（远低于 150），100% early terminal，score -112.64。Agent 从未学会稳定飞行/着陆，一直在快速坠毁。

### 2. 哪个组件是主因？
**`landing_bonus` — 稀疏完成代理的典型案例。** nonzero_rate = 0.19%，即每 500+ 步才触发一次。Agent 几乎从未体验过正奖励。`distance_reward = -dist` 始终为负（mean -0.972），`stability_penalty` 也始终为负（mean -0.144），整个 episode 只有惩罚没有奖励——Agent 学到的唯一策略就是"尽快结束"。

### 3. 上一轮改了什么？
从 `progress_delta`（最高 197 分）切换到 `distance_reward + landing_bonus`。这是回归——`progress_delta` 天然有正有负，而 `distance_reward` 纯负且依赖一个从不触发的 bonus。

---

**修复方向：** `landing_bonus` 是典型的稀疏完成代理问题（`sparse_completion_proxy`）。按知识库建议，将二值条件改为连续 bounded 函数，让 Agent 每一步都能感受到"靠近目标 = 更多奖励"。保持其他组件不变以归因。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 到原点 (0,0) 的距离
    dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 主信号1：负距离——提供指向原点的密集梯度
    distance_reward = -1.0 * dist

    # 主信号2：连续接近奖励——替代稀疏的 landing_bonus
    # 数学形态：bounded 在 [0, 2]，dist=0 时=2.0，dist=1 时≈0.33
    # 每一步都有正向信号，越近越强，解决 nonzero_rate=0.19% 的问题
    proximity_bonus = 2.0 / (1.0 + 5.0 * dist)

    # 稳定约束——保持上一轮系数不变，单独归因本轮改动
    vel_penalty = 0.1 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.2 * abs(next_obs[4])
    angvel_penalty = 0.05 * abs(next_obs[5])
    stability_penalty = -vel_penalty - angle_penalty - angvel_penalty

    total_reward = distance_reward + proximity_bonus + stability_penalty

    components = {
        'distance_reward': distance_reward,
        'proximity_bonus': proximity_bonus,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

**预期效果：** `proximity_bonus` 的 nonzero_rate 将接近 100%，mean 预计在 0.3~0.5 左右（取决于 Agent 初始距离分布）。结合 `distance_reward ≈ -1.0`，净信号仍偏负但在接近原点时转正（dist≈0.4 处盈亏平衡）。这给 Agent 提供了一个可追逐的正向目标，而不是只能"少扣分"。
