## 1. 诊断报告

**evidence**
- `read_memory()`：iter_1（无能量项，`forward_progress=2.0*v/(1+|v|)` + `upright_stability`）得分 **278.99**，为历史最佳；iter_2 加 `-0.03*action_sq` 后降到 274.34；iter_3 把前向增益提到 2.5、能量惩罚降到 -0.01 后大幅降到 **240.53**。
- 反推每步前向速度：
  - iter_1：`697.414 / 1275.6 ≈ 0.547` → `v ≈ 0.376 m/s`
  - iter_2：`720.870 / 1427.0 ≈ 0.505` → `v ≈ 0.338 m/s`
  - iter_3：`861.500 / 1385.9 ≈ 0.622`（但增益为 2.5）→ `v ≈ 0.331 m/s`
- `upright_stability` ep_sum：iter_1 = **-6.678**（几乎不惩罚，说明躯干稳定）；iter_2 = -43.2；iter_3 = **-72.99**。增益从 2.0 提到 2.5 后，前向奖励上限从 2.0 升到 2.5，**超过了直立惩罚上限 2.0**（angle 项最大 1.5 + angvel 项最大 0.5），导致策略用前倾姿态追速度，越练越不稳。
- `read_checkpoint_trend(1/2/3)`：iter_3 的 late-checkpoint 分数（cp80=184.8 → cp100=187.5）与最终 240.53 都不如 iter_1/2，说明增益提高 + 保留能量项的组合是负贡献。
- 结论：能量项两轮都拖慢步态；单纯提高饱和增益会诱发前倾 exploit。真正没被利用的改进空间是**前向奖励的饱和形态**——iter_1 的 `2v/(1+|v|)` 在高速区边际收益快速衰减，策略停在 0.376 m/s。

**behavior_diagnosis**
- iter_1 学到的是稳定但偏慢的步态（0.376 m/s，len≈1276，upright 惩罚几乎为 0），已接近目标但存在速度平台。
- iter_3 学到的是更慢且更不稳定的步态（0.331 m/s，upright=-73，score_range 出现 -7.6 的摔倒样本），这是不健康的，属于“前倾追速度”的 exploit 前兆。

**signal_completeness**
- 主职责 `forward_progress` 存在但饱和过快，缺少高速区梯度。
- 次职责 `upright_stability` 存在且形态有效，iter_1 证明它足以维持稳定。
- `energy_efficiency_penalty` 两轮实证为负贡献，应删除。

**selected_level**
- **L2**。理由是：L1 直接回到 iter_1 配方已被判为重复草稿，不是有效搜索干预；而当前唯一有证据支持的改进是把前向信号从“快速饱和”改为“有界线性”，改变数学形态以突破速度平台。

**selected_intervention**
- 目标组件：`forward_progress`。
- 具体修改：把 `2.0 * horizontal_velocity / (1.0 + abs(horizontal_velocity))` 改为 `1.5 * clip(horizontal_velocity, -1.0, 1.0)`；同时删除 `energy_efficiency_penalty`。
- 关键设计：低速区奖励与 iter_1 接近（v=0.376 时 0.564 vs 0.547），不会降低现有速度激励；高速区保持线性梯度；**上限 1.5 < 直立惩罚上限 2.0**，保留稳定性预算，避免 iter_3 的前倾 exploit。

**falsifiable_hypothesis**
- 线性前向奖励在 0.333 m/s 以上持续给出比饱和形式更高的边际收益，预计推动策略突破 0.376 m/s 平台、走得更快；同时上限 1.5 严格小于直立惩罚上限 2.0，预计 `upright_stability` 不会像 iter_3 那样恶化到 -70 量级，最终 score 应回到并超过 iter_1 的 278.99。

**expected_next_round**
- 策略步态速度高于 0.376 m/s，`forward_progress/len` 升高，`upright_stability` 保持在小幅负值（接近 iter_1 的 -6~-10 量级），`terminated` 仍为 20/20，score 接近或超过 280，向 300 逼近。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：有界线性前向速度奖励。
    # iter_1 的饱和形式 2v/(1+|v|) 在高速区边际收益快速衰减，策略停在 0.376 m/s。
    # iter_3 把增益提到 2.5 后，前向奖励上限 2.5 超过直立惩罚上限 2.0，
    # 诱导躯干前倾追速度（upright_stability 从 -6.7 恶化到 -73）。
    # 这里改为 1.5 * clip(v, -1, 1)：
    #   - 低速区与 iter_1 接近（v=0.376 时 0.564 vs 0.547），不降低现有速度激励；
    #   - 高速区提供持续线性梯度，推动更快步态；
    #   - 上限 1.5 < 直立惩罚上限 2.0，保留稳定性预算，避免前倾 exploit。
    horizontal_velocity = next_obs[2]
    clipped_velocity = max(-1.0, min(1.0, horizontal_velocity))
    forward_progress = 1.5 * clipped_velocity

    # 稳定性约束：惩罚躯干倾角与躯干角速度（保持 iter_1 已验证的形态不变）。
    torso_angle = next_obs[0]
    torso_angle_sq = torso_angle * torso_angle
    angle_penalty = 1.5 * torso_angle_sq / (1.0 + torso_angle_sq)

    torso_angular_velocity = next_obs[1]
    torso_angular_velocity_sq = torso_angular_velocity * torso_angular_velocity
    angular_velocity_penalty = 0.5 * torso_angular_velocity_sq / (1.0 + torso_angular_velocity_sq)

    upright_stability = -(angle_penalty + angular_velocity_penalty)

    # 删除 energy_efficiency_penalty：
    # iter_2(-0.03*action_sq) 与 iter_3(-0.01*action_sq) 都拖慢步态并恶化 upright，
    # 已实证为负贡献，次目标不应压过前进与稳定主目标。
    total_reward = forward_progress + upright_stability

    components = {
        "forward_progress": forward_progress,
        "upright_stability": upright_stability,
    }

    return float(total_reward), components
```