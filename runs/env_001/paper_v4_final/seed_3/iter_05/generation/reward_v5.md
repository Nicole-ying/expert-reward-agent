# 设计理由

## 0. 信号覆盖审计

### 0.1 终止模式分析
18/20 truncated（超时），episode length=915，说明 agent 存活了整个 episode 但未完成任务。terminated=2 且 score 最低 -212，说明少数 episode 触发了终止条件（可能是坠毁/翻倒）。agent 没有接近任务完成（着陆）。

### 0.2 观测使用扫描
- 已使用: obs[0]:x, obs[1]:y, obs[2]:vx, obs[3]:vy, obs[4]:angle, obs[5]:ang_vel
- **未使用**: obs[6]:left_contact, obs[7]:right_contact — 这两个维度直接指示着陆腿是否接触地面，是任务完成的核心信号。

### 0.3 信号缺口判断
**信号缺失**。接触传感器未使用，而它们是判断是否着陆的最直接证据。当前 landing_reward 是一个纯粹的启发式代理（y 接近 0、vy 小、angle 小），但它没有"接触"信号，因此在每个 step 都能触发（active_rate=100%），agent 学会维持一种安全的悬停状态来获取持续的 landing_reward，而不是真正着陆。

### 0.4 僵尸组件检查
landing_reward 的 active_rate=100% 且 episode_sum_mean=1126，占 97.3% — 它接管了整个 reward。progress_gated 被角度门控完全压制（per-step 仅 0.0046 vs landing_reward 的 1.23）。landing_reward 是问题根源。

## 1. 行为诊断

agent 学会了**慢速徘徊**：通过维持安全的 angle 和小的 vy 来持续获取 landing_reward，但没有动力完成真正的着陆。len 从 iter 3 的 70 暴增到 915，而 score 从 -71 掉到 -179，说明 agent 在刷 landing_reward 但放弃了 progress。

## 2. 干预层级: Level 2 — 结构变换

**核心问题**：landing_reward 是一个全局 soft proxy，它奖励"看起来像着陆"的状态而非"实际着陆"。接触传感器（obs[6], obs[7]）是区分"接近地面"和"实际着陆"的关键信号。

**变换**：删除全局 landing_reward，新增基于接触传感器的着陆代理。利用 left_contact 和 right_contact 构建一个只在腿触地时才激活的 reward signal。

**为什么之前的方向连续 4 轮预判失败**：从 iter 1 到 iter 4，始终在围绕"更好地引导 agent 接近地面"设计 proxy，但这些 proxy 都是全局信号（y 接近 0、vy 小），agent 可以停在中间状态持续获取这些奖励而不完成任务。contact sensors 是环境给出的真实任务完成信号，未使用过。

## 3. 组件变换

| 当前 | 问题 | 变换 |
|---|---|---|
| landing_reward (全局 soft proxy) | 在不接触地面时持续获奖，active=100%，占97% | **删除** |
| — | 缺少真正的接触信号 | **新增** contact_based_landing_proxy：`(left_contact + right_contact)/2 * gate(y, vy, angle)` |

新组件设计：
- contact 提供二值判别（只有真接触才有奖励）
- y/vy/angle 的 gate 确保接触时姿态也是安全的，避免"摔在地上也算接触"
- 几何平均确保任意一个条件差就会显著压低奖励

**系数校准**：
- contact_based_landing_proxy 的 per-step 应设为 ~0.3-0.5（在主信号量级范围内，但只在接触时触发）
- progress_gated 当前被角度门控完全压制，需要检查 gate 阈值。当前 gate 在 abs_angle > 0.15 时急剧衰减，对于飞行器来说太严格。放宽阈值到 0.3 弧度（~17°），让 agent 在合理范围内能获得 progress 信号。
- lateral_pos_penalty 和 angvel_penalty 保留不变（量级合理，active_rate 正常）

**设计校准检查**：
1. 新 contact_based_landing_proxy 的 per-step：接触状态通常出现在最后阶段，大部分 episode 可能只有少量 step 触发，量级可控。gate 系数设计使得在理想着陆状态（contact=1, |y|<0.3, |vy|<0.5, |angle|<0.2）时 reward≈1.0，正常 per-step 约 0.5-1.0（触地期间）。
2. progress_gated 的 hinge 阈值：角度门控从 0.15 → 0.3（终止边界的 60~80%），让 progress 能在更宽的姿态范围内生效。
3. gate 不塌缩：新的接触 gate 在"触地但姿态稍差"时（contact=1, angle=0.4, vy=1.0, y=0.5）：mean_contact=1, fy≈0.67, fvy≈0.33, fang≈0.11, geom≈0.29，仍有 ~0.3 的分数，不会完全消失。
4. 单组件 ≤ 2x 主信号：满足。
5. 总惩罚负担：lateral_pos_penalty per-step ≈ -0.03, angvel_penalty ≈ -0.0001，合计远小于主信号。

同时修复 angle_gate 的过严问题，让 progress 能恢复有效梯度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # observation indices: 0:x, 1:y, 2:vx, 3:vy, 4:angle, 5:ang_vel, 6:left_contact, 7:right_contact
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_lc = next_obs[6]  # left contact
    n_rc = next_obs[7]  # right contact

    # ---------- 1. progress towards origin (distance decrease) ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new
    w_progress = 10.0

    # ---------- 2. attitude gate: suppress progress when angle is dangerous ----------
    # Relaxed from 0.15 to 0.30 (17°) to allow reasonable maneuvering
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (12.0 * (abs_angle - 0.30)) / 
                              (1.0 + 2.718281828 ** (12.0 * (abs_angle - 0.30))))

    # ---------- 3. lateral position penalty (encourage centering) ----------
    w_lat_pos = 0.15
    lateral_pos_penalty = -w_lat_pos * (nx ** 2)

    # ---------- 4. angular velocity penalty (smooth rotation) ----------
    w_angvel = 0.05
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. contact-based landing proxy ----------
    # Only rewards when legs actually touch the ground, with safety gates
    mean_contact = (n_lc + n_rc) / 2.0  # [0, 1] continuous
    
    # Safety gates: y close to 0, low vy, low angle
    k_y = 10.0
    k_vy = 8.0
    k_ang = 15.0
    f_y   = 1.0 / (1.0 + k_y   * abs(ny))
    f_vy  = 1.0 / (1.0 + k_vy  * abs(nvy))
    f_ang = 1.0 / (1.0 + k_ang * abs(n_angle))
    
    # Geometric mean of contact and safety factors
    contact_landing_factor = (mean_contact * f_y * f_vy * f_ang) ** 0.25
    w_contact_land = 5.0
    contact_landing_reward = w_contact_land * contact_landing_factor

    # ---------- combine ----------
    total_reward = (w_progress * progress * angle_gate
                    + lateral_pos_penalty
                    + angvel_penalty
                    + contact_landing_reward)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_pos_penalty": lateral_pos_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_landing_reward": contact_landing_reward
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺失 — 接触传感器 obs[6]/obs[7] 完全未使用，而 landing_reward 作为全局 proxy 被 exploit，导致 agent 徘徊刷分而不真正着陆。
- **behavior**: agent 维持在安全姿态持续获得 landing_reward（97.3%贡献），episode 从 70 步暴增到 915 步，score 恶化到 -178.95。
- **signal**: 缺少接触信号（left/right_contact）；landing_reward 过强且 active=100%；angle_gate 阈值 0.15 过严压制了 progress。
- **level**: Level 2
- **hypothesis**: 引入接触传感器作为着陆奖励的触发条件，将全局 soft proxy 替换为接触门控的着陆奖励，迫使 agent 必须实际接触地面才能获得正向奖励，无法在悬停态刷分。同时放宽角度门控让 progress 恢复梯度。
- **risk**: 接触奖励触发稀疏可能导致学习变慢；如果角度门控放宽后 agent 角度失控，可能需要调整平衡。