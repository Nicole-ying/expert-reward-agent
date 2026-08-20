# Response Record

## 诊断

**问题 1：验证失败** — 上一轮 `components` 中包含了 `total_reward`，但指令要求 "components 只放总公式中直接出现的变量"。`total_reward` 是公式本身而非公式中的变量，这是验证失败的直接原因。

**问题 2：势能塑形彻底失败（-1369 vs 历史最佳 -112）**

| 指标 | 历史最佳 (iter 1) | 势能塑形 (iter 3) |
|------|------------------|-------------------|
| score | -111.87 | **-1368.99** |
| len | 68.45 | 177.60 |
| 主信号 | distance_reward=-0.971 | potential_shaping=+0.126 |
| early_terminal | N/A | **75%** |

根因分析：
- `potential_shaping` 的 `abs_contrib_%` 仅 1.96%，而 `original_env_reward` 占 **94%**。我们的塑形信号被环境原始奖励完全淹没。
- 势能函数同时包含 speed 和 angle，agent 可能学会"原地悬停减小 speed/angle"来获取正奖励，而非真正飞向目标。
- 75% 的 episode 提前终止（crash），说明塑形信号没有提供有效的避撞引导。

**修复方案：回到已验证有效的 `distance_reward` 骨架，应用 `bounded_continuous_proxy` 技法改进 landing 信号。**

- 保留 `distance_reward`（主学习信号）
- 保留 `stability_penalty`（弱背景约束）
- 将二值 `soft_landing_proxy` 改为连续乘积形式（`near_factor × slow_factor × upright_factor × contact_factor`），提供稠密梯度
- 移除 components 中的 `total_reward`

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 主学习信号：负距离，引导飞行器持续靠近原点 ===
    # 这是唯一被验证有效的信号骨架（iter 1: score=-111.87）
    dist_to_target = ((next_obs[0] - 0.0) ** 2 + (next_obs[1] - 0.0) ** 2) ** 0.5
    distance_reward = -dist_to_target

    # === 稳定性惩罚：弱背景信号，抑制高速/大角度/高角速度 ===
    speed_norm = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])

    lambda_vel = 0.02
    lambda_angle = 0.02
    lambda_av = 0.02

    stability_penalty = -(lambda_vel * speed_norm + lambda_angle * angle_abs + lambda_av * angvel_abs)

    # === 软着陆连续代理奖励 ===
    # 改进：从二值 if 条件改为连续乘积因子（bounded_continuous_proxy 技法）
    # 每个因子 ∈ [0,1]，乘积提供稠密梯度，引导 agent 同时满足所有着陆条件
    near_factor = max(0.0, 1.0 - dist_to_target / 0.5)       # dist<0.5 时 >0，越近越大
    slow_factor = max(0.0, 1.0 - speed_norm / 0.3)           # speed<0.3 时 >0，越慢越大
    upright_factor = max(0.0, 1.0 - angle_abs / 0.1)         # angle<0.1 时 >0，越正越大
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0       # 平均接触 [0,1]

    soft_landing = near_factor * slow_factor * upright_factor * contact_factor * 5.0

    # === 总奖励 ===
    total_reward = distance_reward + stability_penalty + soft_landing

    components = {
        'distance_reward': distance_reward,
        'stability_penalty': stability_penalty,
        'soft_landing': soft_landing,
    }

    return float(total_reward), components
```
