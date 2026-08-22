# Response Record

# 设计理由

**第0步审计**：当前骨架全由惩罚驱动——lateral_penalty 与 upright_penalty 几乎完全支配奖励（episode_sum_mean 为 2.05 和 0.24，主信号 gated_forward 仅 2.02），导致 agent 在短短几步内因高度越界终止（len=9.95，terminated=20/20）。根本原因是惩罚过强且无正向恢复梯度。

**Level 2 结构变换**：将 `lateral_penalty`（独立惩罚项）改造为 **横向稳定性门控因子** 乘入前进奖励。横向速度不再是扣分项，而是仅当它过大时才削弱前进奖励的 scale factor，这样 agent 永远不会因单纯侧移而受到纯负奖励，只在试图前进时被提醒保持直线。新的 lateral_gate 有界有梯度，避免了早期探索被惩罚扼杀。保留 upritght_penalty（其 per-step 已很小），但横向惩罚的移除大幅降低了惩罚总负担，使主信号能够主导学习方向。

**为什么过去连续失败**：从 iter 5 起每次迭代都带有独立强惩罚项（lateral_penalty、upright_penalty、joint_vel_penalty 等），而主前进信号都被惩罚压制。agent 学到的策略是尽快终止以累积最小负奖励，而不是探索前进。本次变换从结构上消除这一负反馈螺旋。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next_obs
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]

    # Body uprightness (1.0 = perfectly upright, 0.0 = tilted)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    body_up_z = max(0.0, min(1.0, body_up_z))

    # Forward progress (bounded, only positive velocity)
    vx = max(0.0, body_x_vel)
    forward_reward = vx / (1.0 + vx)          # bounded in [0, 1)

    # Height safety gate
    low_gate  = max(0.0, min(1.0, (body_z - 0.2) / 0.1))   # 0 at 0.2, 1 at 0.3
    high_gate = max(0.0, min(1.0, (1.0 - body_z) / 0.1))   # 0 at 1.0, 1 at 0.9
    height_gate = low_gate * high_gate                     # 1 inside safe zone

    # Lateral stability gate (replaces lateral_penalty)
    # Gate decays smoothly with absolute lateral velocity, never goes negative
    lateral_gate = 2.718281828 ** (-abs(body_y_vel) / 0.5)  # ~1 at vy=0, ~0.37 at vy=0.5

    # Upright posture penalty (preserved but will be dominated by forward)
    upright_penalty = (1.0 - body_up_z)**2

    # Weights
    w_forward  = 1.0
    w_upright  = 5.0

    total_reward = (w_forward * height_gate * lateral_gate * forward_reward
                    - w_upright * upright_penalty)

    components = {
        "gated_forward":   w_forward * height_gate * lateral_gate * forward_reward,
        "upright_penalty": w_upright * upright_penalty,
        "lateral_gate":    lateral_gate   # factor, for monitoring
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前 reward 漏了正向稳定性信号，完全依赖惩罚，导致 agent 在早期就被负奖励淹没
- **behavior**: agent 以最快速度终止（len=9.95），纯粹为了最小化累积惩罚
- **signal**: 横向移动被过度惩罚（per-step 0.206，是主前进信号的 1.01 倍），缺少安全探索的梯度
- **level**: Level 2
- **hypothesis**: 将 lateral_penalty 改为门控因子可消除纯负反馈螺旋，让主前进奖励主导学习，agent 将延长生存并开始探索直线前进
- **risk**: 若 height_gate 仍过早关闭，agent 可能在安全区域边缘仍有短暂惩罚主导期，但横向门控已排除主要负项，风险可控
