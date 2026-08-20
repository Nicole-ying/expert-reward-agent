# Response Record

# 设计理由

## 第 0 步：信号覆盖审计

a) **终止 → 前兆**：#5 §5 声明了两个终止条件：(1) 身体高度超出 (0.2, 1.0)；(2) NaN/inf。当前代码中，高度有 `height_penalty`（hinge at 0.25/0.95），但 active_rate 仅 2.5%——门阈值设得太紧或形式有误，几乎从不触发。直立信号（`upright_penalty`）和高度没有直接关联，但 upright 在 100% 步触发，毫无区分度。**结论：高度前兆信号几乎缺失，直立前兆无梯度。**

b) **目标 → 进度**：#5 §1 声明目标是"持续向前移动 + 保持健康高度和直立"。当前代码有 `forward` 直接给向前速度梯度，有。但高度和直立只有惩罚，没有正向吸引子。**结论：有进度信号，但缺少对存活状态的正面激励。**

c) **效率信号**：#5 §4 动作维度 = 8 ≥ 6，当前代码无 action penalty。**问题存在但非首要——先解决 59 步就死的问题。**

d) **僵尸组件**：#4 组件表 `height_penalty` active_rate = 2.5% < 2%。应删除或改造。

e) **一句话结论**：当前 reward 漏了"将存活状态转化为可学习梯度"的机制——两个惩罚组件都无法引导 agent 远离终止边界。

---

## 1. 行为诊断

1. **agent 在做什么？** 快速失败。平均 59 步全部 terminated，20/20 提前死亡。score 区间 [-65.46, 19.19]，最差的 2 个 episode 在 150 步前就因 score < -50 被判为 early terminal。

2. **干预哪个目标？** 存活。59 步什么都学不了。核心问题是 agent 感觉不到"接近死亡"和"安全"之间的梯度差异。

3. **方向还值得继续吗？** 第一轮，没有历史包袱。当前骨架（forward + 两个惩罚）逻辑正确但实现不到位。不重建。

---

## 2. 干预层级：Level 2 — 结构变换

**核心问题**：两个惩罚都失效。

| 组件 | 证据 | 问题 | 变换 |
|---|---|---|---|
| `height_penalty` | active_rate 2.5% | 阈值 0.25/0.95 太窄，hinge 几乎不触发 | 放宽阈值到 0.3/0.9，同时改为 gate 而非 penalty |
| `upright_penalty` | active_rate 100%，无动态范围 | 二次形式 `1 - body_up_z` 几乎恒正，无区分度 | 改为 hinge（只惩罚 body_up_z < 0.7），让"基本直立"区域梯度为零 |

**设计选择**：引入 `health_gate` 替代独立 penalty。将高度和直立做成门控因子，乘法作用于 forward reward。这样：
- agent 要获得前进奖励，必须同时保持高度和直立 → 存活不再是"挨罚"，而是"获得奖励的前提"
- gate 在安全区域 ≈ 1，在边界附近平滑下降到 0 → 每步都有梯度引导回安全区

---

## 设计校准

**主信号 per-step**：`forward` ep_sum_mean = 72.31，len = 59.15 → per-step ≈ 1.22。新 forward 系数保持 1.0。

**新 gate 设计**：
- 高度因子：sigmoid-style。在 (0.25, 0.95) 内 ≈ 1，边界外平滑衰减。用 `min(1.0, 2.5 * height_low_factor)` 形式，让 z=0.22 时 gate ≈ 0.8，z=0.20 时 ≈ 0.5。
- 直立因子：hinge at 0.6。body_up_z > 0.6 时 gate=1，<0.6 时线性衰减。
- 两个因子取 `min`保证任一条件恶化都限制奖励。

**惩罚负担**：
- 移除 `height_penalty`（被 gate 替代）。
- `upright_penalty` 改为 hinge：只惩罚 body_up_z < 0.6，系数 0.5。per-step ≈ 0.15~0.3（远低于 forward 的 1.22）。
- 新加 `z_velocity_penalty`：惩罚垂直弹跳（|body_z_velocity| > 1.5），系数 0.2，辅助稳定。

总惩罚 per-step ≤ 0.5，远低于 1.22 的 0.5x 阈值。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- Extract observations ----------
    body_z = obs[0]
    quat_w = obs[1]
    quat_x = obs[2]
    quat_y = obs[3]
    quat_z = obs[4]
    body_x_velocity = obs[13]
    body_z_velocity = obs[15]

    # ---------- Derived signals ----------
    # Uprightness: body_up_z = 1.0 - 2*(qx^2 + qy^2), range ~[-1, 1]
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- Health gate factors ----------
    # Height factor: healthy in (0.25, 0.95), smooth decay outside
    # Below 0.25: linear from 1.0 at z=0.25 down to 0.0 at z=0.15
    height_low_danger = max(0.0, (0.25 - body_z) / 0.10)  # 0.0 when z>=0.25, 1.0 when z<=0.15
    height_low_factor = max(0.0, 1.0 - height_low_danger)
    # Above 0.95: linear from 1.0 at z=0.95 down to 0.0 at z=1.05
    height_high_danger = max(0.0, (body_z - 0.95) / 0.10)
    height_high_factor = max(0.0, 1.0 - height_high_danger)
    height_factor = min(height_low_factor, height_high_factor)

    # Upright factor: healthy when body_up_z > 0.6, linear decay below
    upright_factor = min(1.0, max(0.0, (body_up_z - 0.2) / 0.4))
    # At body_up_z=0.6: (0.6-0.2)/0.4 = 1.0  → gate=1.0
    # At body_up_z=0.2: (0.2-0.2)/0.4 = 0.0  → gate=0.0
    # Smooth gradient throughout

    # Combined health gate: must satisfy both to earn full forward reward
    health_gate = min(height_factor, upright_factor)

    # ---------- Reward components ----------
    # 1) Forward progress gated by health
    forward_progress = max(0.0, float(body_x_velocity))
    reward_forward = 1.0 * forward_progress * health_gate

    # 2) Uprightness hinge penalty (only active when severely tilted)
    # body_up_z < 0.6: penalty grows linearly as tilt increases
    upright_error = max(0.0, 0.6 - body_up_z)
    reward_upright_penalty = -0.5 * upright_error

    # 3) Vertical velocity penalty (discourage bouncing/launching)
    # Soft hinge: penalize |vz| > 1.5
    vz_magnitude = abs(float(body_z_velocity))
    vz_excess = max(0.0, vz_magnitude - 1.5)
    reward_z_velocity_penalty = -0.2 * vz_excess

    # ---------- Total reward ----------
    total_reward = reward_forward + reward_upright_penalty + reward_z_velocity_penalty

    components = {
        "forward_gated": reward_forward,
        "health_gate": health_gate,
        "upright_penalty": reward_upright_penalty,
        "z_velocity_penalty": reward_z_velocity_penalty
    }

    return float(total_reward), components
```

---

# 诊断摘要
- **audit**: 高度前兆信号几乎缺失（active_rate 2.5%），直立惩罚无动态范围（100%），两个惩罚都无法提供存活梯度
- **behavior**: agent 快速失败——20/20 提前终止，平均 59 步，无法学会避免高度越界或严重倾斜
- **signal**: 缺的是"存活状态→可学习梯度"的桥梁；现存的惩罚只是恒定的负偏置，不引导行为改善
- **level**: Level 2 — 结构变换（引入 health_gate 乘法门控 + 直立 hinge 替代无梯度二次惩罚）
- **hypothesis**: health_gate 将"保持安全高度和基本直立"从被动挨罚变成获得 forward 奖励的前提，agent 每步都有梯度驱向安全区；直立 hinge 在 body_up_z > 0.6 时梯度为零，只惩罚严重倾斜，让 agent 在基本直立时有自由探索如何前进的空间
- **risk**: gate 可能偶尔塌缩为 0（尤其在早期探索阶段），导致 forward 奖励消失，agent 可能暂时没有学习信号——但 z_velocity_penalty 和 upright_penalty 在此期间仍然提供弱引导，且随着探索 gate 会逐步恢复
