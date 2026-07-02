## 诊断

**1. Agent 发生了什么？**
Episodes 极短（~73步），100% early terminal。Score ~-110，original_env_reward = -1.53。Agent 在快速坠毁，没有学会靠近目标。

**2. 哪个组件是主因？**
`progress_delta` 均值仅 0.016——信号太弱，无法提供有效学习梯度。上轮把 `stability_penalty` 从 9x 削弱到 37% 解决了比率问题，但得分毫无改善，说明根因不是 penalty 压制，而是 progress 信号本身弱到不足以驱动学习。

**3. 我之前改了什么？**
Iter 1→2：削弱 stability_penalty 权重 25x。成功降低了 penalty 比率，但得分没变。根因是 progress 信号弱，不是 penalty 强。

**结论：** 同一骨架（progress_delta + soft_landing + stability_penalty）已迭代 2 轮，最佳得分 -108.94，远低于任何合理 target 的 25%。按原则 3，应换骨架。

**修改方案：** 将主信号从 `progress_delta` 替换为 `bounded_proximity`（1/(1+k·dist)）。该信号自动 bounded 在 [0,1]，靠近目标时自然增长，始终为正（鼓励存活），且梯度平滑。保持 stability_penalty 轻量权重和 soft_landing_bonus 不变。

同时修复可能的 validation 问题：移除 components 中冗余的 `total_reward` 键，清理未使用变量。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 主学习信号：bounded proximity ===
    # 替代 progress_delta。1/(1+k*dist) 自动 bounded 在 [0,1]，
    # 靠近目标时自然增长，始终为正（鼓励存活），提供平滑梯度。
    x_next, y_next = next_obs[0], next_obs[1]
    d_next = (x_next ** 2 + y_next ** 2) ** 0.5
    k_proximity = 5.0  # k=5: dist=1→0.167, dist=0.5→0.286, dist=0.1→0.667
    bounded_proximity = 1.0 / (1.0 + k_proximity * d_next)

    # === 稳定约束（保持上轮轻量权重） ===
    x_vel, y_vel = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    w_vel = 0.004
    w_angle = 0.02
    w_angvel = 0.004
    stability_penalty = (
        w_vel * (abs(x_vel) + abs(y_vel)) +
        w_angle * abs(body_angle) +
        w_angvel * abs(ang_vel)
    )

    # === 软着陆奖励（保持原有逻辑） ===
    near = (d_next < 0.1)
    slow = (abs(x_vel) < 0.1 and abs(y_vel) < 0.1)
    level = (abs(body_angle) < 0.1)
    both_feet = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    soft_landing_bonus_raw = 1.0 if (near and slow and level and both_feet) else 0.0
    soft_landing_bonus = 0.5 * soft_landing_bonus_raw

    total_reward = bounded_proximity - stability_penalty + soft_landing_bonus

    components = {
        "bounded_proximity": bounded_proximity,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
    }

    return float(total_reward), components
```