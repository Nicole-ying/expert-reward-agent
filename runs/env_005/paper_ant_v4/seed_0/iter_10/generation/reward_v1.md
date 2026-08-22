# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ----- extract next state signals -----
    body_z = next_obs[0]
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    body_z_vel = next_obs[15]
    roll_vel = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel = next_obs[18]

    # ----- derived postural signals -----
    # body up direction: 1 = fully upright, 0 = horizontal, <0 = inverted
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(0.0, min(1.0, body_up_z))   # clamp to [0, 1] (inverted gives 0)

    # height health factor: 0 at lower bound 0.2, 1 at upper bound 1.0
    height_factor = (body_z - 0.2) / 0.8
    height_factor = max(0.0, min(1.0, height_factor))

    # upright factor: 0 at fully horizontal/inverted, 1 at fully upright
    upright_factor = body_up_z

    # ----- multiplicative progress reward with soft health gate -----
    # forward speed is multiplied by height and upright factors:
    #   - if body_z is low or robot is tilted, forward reward is scaled down
    #   - fully healthy posture enables the full forward speed reward
    # scaling constant controls the average per-step reward magnitude
    forward_reward = 3.0 * body_x_vel * height_factor * upright_factor

    # ----- additive stabilisation/constraint penalties -----
    # lateral drift penalty (suppress sideways motion)
    lateral_penalty = -1.0 * (body_y_vel ** 2)

    # angular velocity penalty (suppress excessive roll/pitch/yaw)
    angular_penalty = -0.2 * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    # vertical velocity penalty (suppress bouncing, encourage smooth gait)
    vertical_penalty = -0.5 * (body_z_vel ** 2)

    # ----- total reward -----
    total_reward = forward_reward + lateral_penalty + angular_penalty + vertical_penalty

    components = {
        'forward_reward': forward_reward,
        'lateral_penalty': lateral_penalty,
        'angular_penalty': angular_penalty,
        'vertical_penalty': vertical_penalty,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与角色选择

- **task_family**: `locomotion_continuous_control`  
- **dynamics_subtype**: multi-legged body locomotion (四足，8个力矩控制关节)  
- **primary objective**: 持续向前高速移动  
- **secondary objectives**: 维持身体高度健康、保持直立、抑制侧向漂移与过度抖动

从可用信号与任务画像出发，本次设计的核心角色分解如下：

| role | 性质 | 信号 | formula operator | 理由 |
|------|------|------|------------------|------|
| 前进进度奖励 | **mandatory** (主学习信号) | `body_x_velocity`, `body_z`, `quat` | 连续乘性健康门控 (`soft_health_gate` 变体) | 只有在机器人保持高度与直立时全效给予前进速度奖励，迫使 agent 先稳定再提速 |
| 侧向漂移抑制 | conditional | `body_y_velocity` | `quadratic_penalty` | 抑制侧向漂移，提高前进效率 |
| 角速度稳定性 | conditional | `roll_vel, pitch_vel, yaw_vel` | `quadratic_penalty` | 抑制剧烈抖动与翻转趋势 |
| 垂直方向平滑 | conditional | `body_z_velocity` | `quadratic_penalty` | 减少跳跃，鼓励平稳步态 |

**排除的角色及原因**：
- `terminal_success_reward`：环境无显式成功标志，且 `info` 不可用 → 排除。
- `terminal_failure_penalty`：环境无显式失败标志，且 `info` 不可用 → 排除。
- `action_cost` / `energy_efficiency`：v1 阶段优先学会任务方向，能耗优化留到后续迭代。
- `action_smoothness_penalty`：无 `previous action` 或历史信息 → 不可用。

## 为何放弃旧尝试的“加性惩罚 + 速度奖励”结构

过去的试验（如 `forward_reward + height_penalty + upright_penalty + lateral_penalty + angular_penalty`）最佳仅达到 203.9 分，远未达标。主要瓶颈在于：
- 加性惩罚只在高偏差区域给出强梯度，死区大，无法密集引导 agent 在早期探索中同时保持稳定和提速。
- 速度奖励与健康约束相互独立，agent 容易在速度项获得高奖励而忽略姿态，导致翻倒后学习停滞。

## 新设计机制：乘性健康门控

- **数学结构**：`forward_reward = w * body_x_vel * height_factor * upright_factor`
- **height_factor** 在身体高度自 0.2 至 1.0 线性增长，**upright_factor** 等于 `body_up_z`（下截止为 0）。
- 当身体高度偏低或倾斜较大时，前进奖励被自动衰减，自然将“先恢复健康姿势”内化为获得高速度的前提条件，不需要额外的人为调参。
- 与 soft_health_gate 中的 **乘性形式** 一致，但更为连续且没有硬阈值，保留了密集梯度。
- 该结构从根本上改变了奖励的时间信用分配，避免 past attempts 中“高速前进 → 失稳 → 惩罚无法挽回”的恶性循环。

## 训练后应观察的失败模式

- **stand‑still 或 hover**：如果缩放系数过小或健康门控太严格，agent 可能停在原地保持静止状态。
- **velocity burst then fall**：agent 短暂加速后快速翻倒（证明健康门控的上游梯度仍不足）。
- **diagonal walking / lateral drift**：侧向惩罚不足时可能出现侧向偏行策略。
- **excessive bouncing**：垂直速度惩罚过轻时可能出现跳跃式前进，浪费能量且降低平均速度。

## 后续迭代方向

- 引入 动作幅度/能耗 惩罚（`quadratic_penalty` on actions）提升效率。
- 若前向速度的上界已饱和，考虑将线性速度项替换为饱和函数（如 `tanh`）或平方项，以鼓励更高速度。
- 若出现姿态振荡，可增加关节角速度惩罚或动作差分平滑项（需额外历史信息）。