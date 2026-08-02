# Response Record

# 设计理由

本轮对 **landing_quality** 组件进行结构变换：放宽位置对齐因子的有效半径，让着陆梯度在更早的阶段激活，提供连续、稳定的引导，同时保留最终着陆阶段对高精度的要求。

### 诊断过程

- **审计（§0）**：所有观测维度均被使用，无信号缺口；终止模式为 `terminated`，episode 长度 363，landing_quality 在 12.3% 的步数中贡献 45.6% 的正奖励，说明 agent 已能着陆但只在极短的时间窗内触发了着陆信号。这正是校准问题：着陆信号的触发范围过窄，导致有效梯度稀疏，agent 不易学习到平稳渐近的着陆策略。
- **累积记录**：iter 3 通过放大 progress 系数将分数从 96 拉至 170，但 progress 的引力无法替代着陆阶段的精细控制。iter 2 曾尝试连续化 landing_quality 但失败，病因是当时 progress 太弱（w=5.0），现在 progress 主导（w=20.0），扩大着陆信号范围的安全性已提升。
- **组件异常**：`landing_quality` 的 active_rate 仅 12.3%，反映当前 `altitude_factor` 和 `align_factor` 的容忍半径（分母 0.2）过于苛刻。agent 在满足这些阈值前得不到关于接近程度的反馈，导致着陆缓冲阶段信号缺失，最终速度控制不佳（`landing_velocity_penalty` -6.96）。

### 修改内容：放宽 landing_quality 中位置因子的容忍半径

- `altitude_factor` 的分母从 0.2 → 0.5，`align_factor` 的分母从 0.2 → 0.5。这使 agent 在距离平台垂直/水平 0.5 单位以内时就开始获得非零因子，梯度延伸至着陆前更早的阶段。
- 保持几何平均形式（`product ** (1/6)`），防止乘积在某个因子为 0 时塌缩，但扩大非零区域降低塌缩概率。
- 不调整其他因子及权重，保持最终着陆所需的精度要求不变。
- 单组件 per-step 不超过主信号 2x（估算：放宽后 active_rate 预计升至 20‑30%，landing_quality 均值可能略有增加，但远低于 progress 的 20x 权重产生的主信号）。

### 系数校准

- `w_progress = 20.0`（不变），主信号 per-step ≈ 0.058。
- `w_landing = 2.0`（不变），geo‑mean 形式的输出天然有界（0~1），乘积放缩后最大约 2.0，不足以淹没主信号。
- 惩罚项系数不变，总惩罚负担 < 0.5× 主信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆分
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft_contact = next_obs[6]
    nright_contact = next_obs[7]

    w_progress = 20.0
    w_landing = 2.0
    w_land_vel = 10.0
    w_angle = 0.5
    w_angvel = 0.5
    engine_cost = 0.02

    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5

    # 1. 距离改进
    progress = w_progress * (dist - ndist)

    # 2. 着陆质量软信号 —— 放宽位置因子半径（0.2→0.5）
    altitude_factor = max(0.0, 1.0 - abs(ny) / 0.5)
    align_factor    = max(0.0, 1.0 - abs(nx) / 0.5)
    vx_factor       = max(0.0, 1.0 - abs(nvx) / 0.3)
    vy_factor       = max(0.0, 1.0 - abs(nvy) / 0.5)
    angle_factor    = max(0.0, 1.0 - abs(nangle) / 0.2)
    contact_factor  = 0.1 + 0.9 * (nleft_contact + nright_contact) * 0.5

    product = (altitude_factor * align_factor * vx_factor *
               vy_factor * angle_factor * contact_factor)
    if product > 0.0:
        landing_quality = w_landing * (product ** (1.0 / 6.0))
    else:
        landing_quality = 0.0

    # 3. 着陆速度惩罚（仅在双腿接触时）
    fcontact = float(nleft_contact * nright_contact)
    if fcontact > 0.5:
        vel_pen = -w_land_vel * (nvx**2 + nvy**2)
    else:
        vel_pen = 0.0

    # 4. 姿态稳定惩罚（全程）
    att_penalty = -w_angle * (nangle**2) - w_angvel * (nangvel**2)

    # 5. 引擎使用惩罚
    eng_pen = -engine_cost if action != 0 else 0.0

    total_reward = progress + landing_quality + vel_pen + att_penalty + eng_pen
    components = {
        "progress": progress,
        "landing_quality": landing_quality,
        "landing_velocity_penalty": vel_pen,
        "attitude_penalty": att_penalty,
        "engine_cost": eng_pen
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号覆盖齐全，无关键维度缺失；终止模式以成功着陆为主，但着陆信号触发半径过窄导致稀疏。
- **behavior**: agent 能持续接近平台并着陆，但着陆前缺乏早期梯度引导，缓冲阶段控制不精细。
- **signal**: landing_quality 只在极小邻域内激活（active 12.3%），位置因子阈值过于苛刻，缺少渐进式接近奖励。
- **level**: Level 2
- **hypothesis**: 将位置因子容忍半径从 0.2 拓宽到 0.5，使着陆信号提前 2.5 倍范围激活，为 agent 提供连续的接近梯度，从而改善着陆时的姿态与速度控制，提升最终着陆质量并降低速度惩罚，总分可向 200 靠近。
- **risk**: 若半径过大，landing_quality 可能在远离平台的区域产生非零但无意义的低值，干扰 progress 引导；但当前 progress 主导力度足够（20.0），风险可控。最可能的副作用是 active_rate 升高后 landing_quality 的 episode sum 略微增大，但不至于淹没主信号。
