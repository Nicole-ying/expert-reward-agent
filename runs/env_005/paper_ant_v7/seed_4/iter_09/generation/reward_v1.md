# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals
    body_z = obs[0]          # torso height
    quat_x = obs[2]          # quaternion x component
    quat_y = obs[3]          # quaternion y component
    body_x_vel = obs[13]     # forward velocity
    body_y_vel = obs[14]     # lateral velocity
    roll_vel  = obs[16]      # body angular velocities
    pitch_vel = obs[17]
    yaw_vel   = obs[18]

    # Compute uprightness (z-component of body up vector)
    # Assuming unit quaternion and near‑upright the w component ≈ 1.
    uprightness = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    uprightness = max(-1.0, min(1.0, uprightness))

    # Soft health gate – height component
    # Gate opens linearly from 0 at 0.25 m to 1 at 0.35 m.
    height_gate = (body_z - 0.25) / (0.35 - 0.25)
    height_gate = max(0.0, min(1.0, height_gate))

    # Soft health gate – uprightness component
    # Gate opens linearly from 0 at uprightness 0.5 to 1 at 0.9.
    upright_gate = (uprightness - 0.5) / (0.9 - 0.5)
    upright_gate = max(0.0, min(1.0, upright_gate))

    # Combined gate: minimum ensures both health conditions are partially satisfied
    forward_gate = min(height_gate, upright_gate)

    # Main learning signal: gated forward velocity
    w_fwd = 1.0
    forward_reward = w_fwd * body_x_vel * forward_gate

    # Stabilisation: suppress lateral drift
    w_lat = 0.1
    lateral_penalty = -w_lat * (body_y_vel**2)

    # Stabilisation: damp excessive body rotation
    w_ang = 0.05
    angular_penalty = -w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    total_reward = forward_reward + lateral_penalty + angular_penalty

    components = {
        'forward_reward': forward_reward,
        'lateral_penalty': lateral_penalty,
        'angular_penalty': angular_penalty,
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与职责选取

- **task_family**：`locomotion_continuous_control`
- **dynamics_subtype**：`multi_legged_body_locomotion`（四足、力矩驱动）
- **核心目标**：生成平稳的前向速度（`body_x_velocity`），同时维持安全高度（`body_z` ∈ [0.2, 1.0]）与直立姿态。

## 2. 奖励职责与信号映射

| 职责（role） | 信号来源 | 公式算子 | 说明 |
|---|---|---|---|
| **主学习信号** – 持续前进 | `obs[13]`（前向速度） | `dense_state_signal`（线性） × `soft_health_gate`（min‑门控） | 用高度与直立度共同门控，当身体处于危险区（高度<0.25 或倾斜过大）时主奖励衰减，引导 agent 在安全范围内加速前进。 |
| **姿态稳定** – 抑制侧滑 | `obs[14]`（侧向速度） | `quadratic_penalty` | 轻量惩罚，防止侧向漂移演化成跌倒前兆。 |
| **姿态稳定** – 身体晃动 | `obs[16:19]`（角速度） | `quadratic_penalty` | 抑制快速翻滚、俯仰与偏航，帮助步态平稳。 |

**没有选择的职责及其原因**：
- **terminal_success_reward / terminal_failure_penalty**：环境无显式 success/failure flag（`explicit_success_flag_available=false`，`info` 为空），无法可靠判断。
- **独立高度/姿态惩罚**：改用软门控代替硬惩罚，避免强约束让 agent 不敢移动（前几版实验表明强惩罚和乘积门控导致性能极差）。
- **能耗/扭矩代价**：动作维度虽多（8），但 v1 优先让 agent 学会稳定前进，效率优化留到后续迭代。
- **关节角速度/角度惩罚**：没有在步态未建立前引入额外约束，避免抑制探索。
- **复杂乘积、动态课程、关节条件代理**：不符合 v1 简化原则。

## 3. 关键数学设计思想

- **soft_health_gate 使用 `min` 组合高度和直立度**：与乘积相比，`min` 门控在某一维度暂时恶化时不会导致主奖励完全塌缩，保留了恢复梯度；而当两维度同时严重恶化时，门控趋近零，主奖励完全关闭，迫使 agent 优先恢复健康。
- **门控阈值设置**：高度门控从 0.25 m 开始开启（高于终止阈值 0.2 m 留出缓冲），直立度门控从 `uprightness = 0.5` 开始开启（防止轻微倾斜就切断奖励）。这些设计留有安全探索空间，避免 agent 因害怕失去主奖励而驻足不动。
- **惩罚项权重极轻**：侧向和角速度惩罚仅起“方向盘”作用，不会压倒主前进信号，允许早期步态中出现一定的非理想运动，随后逐步收敛。

## 4. 排除了哪些留到后续迭代的内容

- 关节力矩正则化与动作平滑惩罚（维度 8，留待 v2 提高能效）。
- 更复杂的姿态门控（如单独偏航旋转门）或步态对称性指标。
- 基于 `training_progress` 的动态加权或课程。
- 联合条件完成的软代理（如接近静止 + 低倾倒 + 低扭矩）。

## 5. 预期观察的 failure modes

- **静止或极低速**：若主奖励 + 惩罚总和不足以超过零奖励，agent 可能停步。需观察 forward_reward 的 episode_mean 是否显著大于零。
- **间歇性冲刺后跌倒**：门控在高度/直立度急剧恶化时关闭主奖励，但 agent 仍可能因惯性冲高速度后失稳。需检查 terminated rate 是否随 forward_reward 增加而同步下降。
- **侧向或旋转漂移**：惩罚系数可能过小，若侧向速度或角速度持续增长，可在 v2 中微调系数或改用 hinge 形式。
- **门控塌缩**：如果 agent 始终无法同时满足高度﹥0.25 和直立度﹥0.5，forward_gate 长期为零，主奖励缺失，训练停滞。此时应检查初始条件与随机策略是否足以产生高于 0.25 的高度；必要时可降低高度门控的起始阈值或改用连续函数平滑过渡。

该设计采用与历史尝试中“forward_upright_product + height_survival_penalty”完全不同的机制：以最小门控替代乘积，放弃独立高度惩罚，仅用门控施加生存压力，期望提高主信号的浓度与梯度稳定性。