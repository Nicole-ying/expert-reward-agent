## 诊断

### 1. Agent 的状态
得分从 -89 跳到 242，episode length 从 70 跳到 482，这是质变——dist_gate 的引入让 agent 不再一上来就 crash。当前 agent 已经能稳定存活并完成着陆，得分超过 200 的 solved 阈值。

### 2. 组件分析
- **landing_proxy** mean=0.447，ratio_to_progress=143x。这是总奖励的绝对主导，progress_reward 几乎可以忽略。
- **progress_reward** mean=0.003，ratio=1.0（锚点）。太弱了——当 agent 远离着陆区（dist>1.0，near_factor=0，landing_proxy=0）时，唯一的方向性信号只有这 0.003/step，几乎为零。
- **stability_penalty** mean=-0.001，ratio=-0.33，配合 dist_gate 工作良好，不需要动。
- **dist_gate** 是调制器而非独立奖励信号，ratio 无参考意义。

### 3. 根因
landing_proxy 的 product 形式在远处（near_factor=0）梯度为零，此时唯一的方向引导是 progress_reward，但它弱了 143 倍。Agent 虽然最终学会了着陆（可能靠 landing_proxy 的近场梯度），但 progress_reward 太弱意味着 agent 在远处的探索效率低，接近阶段缺乏清晰的"往哪走"信号。

### 4. 本轮修改
**层次 1：改系数。** 将 `progress_reward` 的系数从 1.0 提升到 5.0。这是一个保守但明确的改动——让主方向信号的强度提升 5 倍，预计 mean 从 0.003 → ~0.015，ratio 从 143:1 → ~30:1。landing_proxy 仍占主导（这是对的，终端状态应该比过程重要），但 progress 不再是可以忽略的背景噪音。

只改这一个地方。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Calculate distance to target (0,0) for obs and next_obs
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Primary learning signal: progress towards the landing pad.
    # Coefficient increased from 1.0 to 5.0 because progress_reward was
    # 143x weaker than landing_proxy (mean 0.003 vs 0.447), making
    # direction-finding nearly blind when far from the pad.
    progress_reward = 5.0 * (dist_obs - dist_next)

    # Stability penalty with distance-gating.
    # Gate = 1/(1+5*dist): ~1.0 at pad, ~0.17 at dist=1.0, ~0.05 at dist=4.0.
    # This lets the agent move freely when far away, and only tightens
    # speed/angle/angvel constraints as it approaches the landing zone.
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    dist_gate = 1.0 / (1.0 + 5.0 * dist_next)
    stability_penalty = dist_gate * (
        -0.01 * speed_next
        - 0.01 * abs(next_obs[4])
        - 0.005 * abs(next_obs[5])
    )

    # Continuous landing proxy: product of bounded factors provides gradient
    # throughout approach. Each factor max(0, 1 - value/threshold) decays
    # linearly from 1→0 as the dimension worsens.
    near_factor = max(0.0, 1.0 - dist_next / 1.0)
    slow_factor = max(0.0, 1.0 - speed_next / 1.0)
    upright_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.5)
    leg_factor = (next_obs[6] + next_obs[7]) / 2.0

    landing_proxy = near_factor * slow_factor * upright_factor * leg_factor

    total_reward = progress_reward + stability_penalty + landing_proxy

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'dist_gate': dist_gate,
        'landing_proxy': landing_proxy,
        'total_reward': total_reward
    }
    return float(total_reward), components
```