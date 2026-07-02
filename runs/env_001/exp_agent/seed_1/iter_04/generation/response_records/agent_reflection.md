# Response Record

## 诊断

### 1. Agent 状态
从 iter 2 的 `progress_delta` 切换到 iter 3 的 `bounded_proximity` 后，得分从 -110 暴涨到 **+251.94**，episode 长度从 73 → 363。Agent 已经学会了基本的降落行为，不再坠毁。

### 2. 组件分析
| 组件 | mean | ratio | nonzero | 评估 |
|------|------|-------|---------|------|
| `bounded_proximity` | 0.56 | 0.56 | 100% | 主信号健康，反向推算平均距离 ~0.157 |
| `soft_landing_bonus` | 0.178 | 0.178 | 35.5% | 触发率不错，但是**二值条件**（4 个 if 全部满足才给 1）→ 梯度稀疏 |
| `stability_penalty` | 0.003 | 0.003 | 100% | 背景约束，量级合理，无需改动 |

### 3. 核心问题
`soft_landing_bonus` 是二值条件：`near AND slow AND level AND both_feet` → 1，否则 → 0。虽然 35.5% 触发率不算低，但**二值信号在条件边界附近没有梯度**——agent 无法感知"我离满足条件还差一点"。搜索结果显示 `bounded_continuous_proxy` 技法正是针对此问题：用连续乘积因子替代 if 条件，每个因子用 `max(0, 1-x/D)` 提供平滑梯度。

### 4. 本轮行动
**层次 2 — 改数学形式**：将 `soft_landing_bonus` 从二值 AND 改为连续乘积，每个因子独立 bounded 在 [0,1]。阈值从 0.1 放宽到 0.2（双倍），确保梯度覆盖更广的区域。`feet_factor` 改用均值而非 min，让单脚触地也能获得部分信号。权重保持 0.5 不变，隔离数学形式变化的效果。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 主学习信号：bounded proximity（不变） ===
    x_next, y_next = next_obs[0], next_obs[1]
    d_next = (x_next ** 2 + y_next ** 2) ** 0.5
    k_proximity = 5.0
    bounded_proximity = 1.0 / (1.0 + k_proximity * d_next)

    # === 稳定约束（不变） ===
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

    # === 软着陆奖励：二值 → 连续乘积（核心改动） ===
    # 每个因子用 max(0, 1-x/D) 提供平滑梯度，阈值 0.2（原二值阈值 0.1 的双倍）
    near_factor = max(0.0, 1.0 - d_next / 0.2)
    speed = abs(x_vel) + abs(y_vel)
    slow_factor = max(0.0, 1.0 - speed / 0.2)
    level_factor = max(0.0, 1.0 - abs(body_angle) / 0.2)
    # 足部接触：均值替代 min，单脚着地也给部分信号（0→0, 单脚→0.5, 双脚→1）
    feet_factor = 0.5 * (next_obs[6] + next_obs[7])

    soft_landing_raw = near_factor * slow_factor * level_factor * feet_factor
    soft_landing_bonus = 0.5 * soft_landing_raw  # 权重保持 0.5

    total_reward = bounded_proximity - stability_penalty + soft_landing_bonus

    components = {
        "bounded_proximity": bounded_proximity,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
    }

    return float(total_reward), components
```
