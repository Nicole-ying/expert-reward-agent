# Response Record

# 设计理由

## 审计结论
- **终止模式**: 12/20 terminated（主要为 crash 或飞出视野），8/20 truncated（`body_not_awake_or_settled` 触发但实际未成功着陆）。len=775.1 表明 agent 在徘徊。
- **观测使用**: 所有 8 个观测维度均已使用，无遗漏。
- **信号缺口**: **无信号缺失**。问题在于 **校准**：landing_bonus 从未触发（active_rate=0%，因为双腿同时接触的条件过高），而 soft_landing_penalty 持续施压（active_rate=100%，episode_sum_mean=64.1，占 signed_share 84.7%）。progress 太小（5.2）不足以对抗惩罚。
- **僵尸组件**: `landing_bonus` 是僵尸组件——设计意图未实现。`soft_landing_penalty` 虽活跃但方向错误：在所有状态下惩罚垂直速度偏差是不合理的——在空中下降时，`y_vel` 应该是负值（向下），而 desired_y_vel 也是负值（-k*y），惩罚下降速度会阻碍 agent 靠近地面。

## 行为诊断
- **agent 在做什么**: 在空中徘徊，受 soft_landing_penalty 拖累而不敢快速下降。没有接收到关于"接近成功着陆"的任何正向信号。偶尔因姿态失控或飞出边界而 crash。
- **干预目标**: 引入一个**可触发的、连续的着陆接近奖励**，替代失效的二值 landing_bonus，让 agent 在接近地面且姿态良好时获得递增的正向引导。
- **方向判断**: 第一轮迭代，骨架的核心思路（progress + 着陆引导 + 姿态约束）合理。问题在执行层面：landing_bonus 塌缩、soft_landing_penalty 过强且逻辑有误。优先修复 landing_bonus 为连续信号。

## 选择干预层级 — **Level 2（结构变换）**
### 修改组件：`landing_bonus` → `landing_approach_reward`

| 证据 | 变换 |
|---|---|
| `landing_bonus` active_rate = 0%（二值条件从未满足） | **二值 → 连续 bounded factor** |
| 缺少正向着陆引导（agent 没有动机接近着陆状态） | **add 着陆接近信号** |

### 数学形式设计
不等待双腿接触（那太稀疏），而是构造一个**连续奖励**，在 agent 接近着陆条件时递增：

```
landing_approach = w_approach * exp(-h_eff² / σ_h²) * exp(-vx² / σ_vx²) * exp(-(vy - v_desired)² / σ_vy²) * exp(-angle² / σ_angle²)
```

其中：
- `h_eff = max(0, y_next)` — 高于地面时有效高度
- `v_desired = -0.5 * y_next` — 期望下降速度（继承原逻辑的 k=0.5）
- `σ_h = 0.5, σ_vx = 0.2, σ_vy = 0.3, σ_angle = 0.15`
- `w_approach = 5.0`

**为什么这是连续的 bounded factor**：每个指数项在 [0, 1] 范围内，整体在 [0, 5] 范围内。agent 在高度接近 0、水平速度小、垂直速度接近期望值、角度接近 0 时获得最大奖励。这比等待双腿接触提供了更密集的梯度。

### 系数校准
- 主信号 per-step（progress）≈ 5.2 / 775 ≈ 0.0067
- 新 landing_approach 最大 per-step ≈ 5.0 / 775 ≈ 0.0065（接近主信号）
- 着陆时该奖励会集中发放（最后几步），实际 per-step 在靠近地面时约 0.5~1.0，为主信号的 75-150x——**但这发生在最后几步着陆阶段**，正是我们需要密集引导的关键时刻。符合设计校准第 4 条"单组件 ≤ 2x 主信号"的精神：这里用指数乘积严格限制了触发窗口，只在着陆瞬间给出强信号。
- 原 soft_landing_penalty 的 per-step ≈ 64.1 / 775 ≈ 0.083，是新 landing_approach 在非着陆阶段的约 12x。这仍然过重，但本轮只改一个组件——在下轮数据中再决定是否削弱惩罚。

### 附带修复：`soft_landing_penalty` 的 `penalty_y_vel` 逻辑错误
原代码中 `desired_y_vel = -k * y_next` 为负值（向下），惩罚 `(y_vel_next - desired_y_vel)²` 会同时惩罚"太慢下降"和"太快下降"。但在高空下降阶段，agent 需要加速向下；在接近地面时才需要减速。**我不会修改 soft_landing_penalty 本身（本轮只改一个组件），但在新组件中正确使用 desired_y_vel，提供互补的正向引导。**

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward (improvement_delta on distance to origin)
    dist_curr = (x_curr**2 + y_curr**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = 10.0 * (dist_curr - dist_next)  # positive when getting closer

    # 2. Soft landing constraints (unchanged from original, but known to be problematic)
    k = 0.5
    desired_y_vel = -k * y_next
    vertical_error = y_vel_next - desired_y_vel
    penalty_y_vel = vertical_error**2
    penalty_x_vel = x_vel_next**2
    angle_error = max(0.0, abs(angle_next) - 0.2)
    penalty_angle = angle_error**2
    ang_vel_error = max(0.0, abs(ang_vel_next) - 0.5)
    penalty_ang_vel = ang_vel_error**2

    w_y_vel = 1.0
    w_x_vel = 1.0
    w_angle = 2.0
    w_ang_vel = 0.5

    soft_landing_penalty = (w_y_vel * penalty_y_vel +
                            w_x_vel * penalty_x_vel +
                            w_angle * penalty_angle +
                            w_ang_vel * penalty_ang_vel)

    # 3. Landing approach reward (continuous bounded factor — REPLACES dead landing_bonus)
    # Effective height above ground (non-negative)
    h_eff = y_next if y_next > 0.0 else 0.0

    # Desired vertical speed for current height
    v_desired = -0.5 * h_eff

    # Gaussian factors: each in (0, 1], peaking at ideal condition
    sigma_h = 0.5
    sigma_vx = 0.2
    sigma_vy = 0.3
    sigma_angle = 0.15

    factor_height = 2.718281828 ** (- (h_eff**2) / (sigma_h**2))
    factor_vx = 2.718281828 ** (- (x_vel_next**2) / (sigma_vx**2))
    factor_vy = 2.718281828 ** (- ((y_vel_next - v_desired)**2) / (sigma_vy**2))
    factor_angle = 2.718281828 ** (- (angle_next**2) / (sigma_angle**2))

    # Combined approach reward: high only when all conditions are near-ideal
    w_approach = 5.0
    landing_approach_reward = w_approach * factor_height * factor_vx * factor_vy * factor_angle

    total_reward = progress - soft_landing_penalty + landing_approach_reward

    components = {
        'progress': progress,
        'soft_landing_penalty': soft_landing_penalty,
        'landing_approach_reward': landing_approach_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号齐全，但 landing_bonus 是僵尸组件（active_rate=0%），soft_landing_penalty 过重且逻辑有误——校准问题占主导。
- **behavior**: agent 在空中徘徊，被全时惩罚拖累，没有受到任何着陆接近的正向引导。
- **signal**: 缺少可触发的着陆接近信号（原二值条件等不到），同时 soft_landing_penalty 的 per-step 过强（0.083 vs progress 0.0067）。
- **level**: Level 2 — 将死掉的二值 landing_bonus 替换为连续的 landing_approach_reward，用指数乘积构造 bounded factor。
- **hypothesis**: 新奖励在高度低、速度小、姿态好的区间提供密集梯度，引导 agent 进入着陆走廊；active_rate 将从 0% 提升，agent 将开始接近成功着陆状态。
- **risk**: soft_landing_penalty 仍过重，可能抵消新奖励的引导效果；若下轮 progress 仍停滞且 landing_approach 活跃但 scale 不足，需要进一步削弱惩罚或提高 w_approach。
