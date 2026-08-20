# 设计理由

## 第 0 步：信号覆盖审计

### 0.1 终止模式分析
训练反馈显示 terminated=20/20，truncated=0/20，80% 的 episode 在 150 步内提前终止（score < -50）。环境事实 §5 声明的终止条件包括：y < 0（触地）、x 超出 [-2.5, 2.5]（超出视口）、angle > 0.5 rad（倒塌）。结合 len=101.6 和极低的 landing_bonus 触发率（0.3%），agent 的主要终止方式是**快速飞出视口边界或坠地**，而非在着陆平台上降落。

### 0.2 观测使用扫描
代码使用了所有 8 个观测维度（0-7），包括 x, y, vx, vy, angle, ang_vel, left_contact, right_contact。没有未使用的观测维度 → **信号齐全**。

### 0.3 信号缺口判断
不存在信号缺失。问题是**信号校准**：progress 组件对"接近目标"的方向不加区分——agent 快速下坠（vy 负大）时 dist_new < dist_old，progress 为正，鼓励向地面俯冲。landing_bonus 的 gate 条件过严（abs(angle)<0.1 且 abs(nvy)<0.2），agent 无法随机满足这些条件，因此无法发现成功的 landing 模式。

### 0.4 僵尸组件检查
landing_bonus active_rate = 0.3%，属于僵尸组件。但此次保留它，因为结构正确——问题在于 agent 从未达到其触发条件需要的能力水平。先修复主力信号引导 agent 到达触发区域，再评估是否调整 gate。

---

## 1. 行为诊断
Agent 的行为是**快速坠毁**：progress 为正（每步约 0.12）但整个 episode 仅持续约 100 步。agent 学会了用最快方式减少距离——俯冲向地面或飞出视口。stability 和 lateral_drift 惩罚太小（合计每步约 -0.05），无法对抗 progress 对"快速下降"的奖励。

**干预目标**：阻止"快速下降触发 progress 奖励"的 exploit。需要在 vy 方向惩罚快速下降速度，让 progress 仅对"受控下降"产生正向收益。

**方向评估**：第一轮，不存在连续预判失败，继续 Level 1 修复。

---

## 2. 干预层级：Level 1 — 尺度修复
当前职责已完备，但 progress 的 magnitude（episode_sum_mean=12.45）远超惩罚（合计约 -5.0），导致 agent 忽视约束。

**修复方案**：

1. **新增：下降速度惩罚（descending_penalty）**
   - 原因：当前 progress 对任何形式的距离减少都给予奖励，agent 通过快速下坠获得大量 progress。需要在 vy 负方向增加抵抗。
   - 数学形式：hinge 惩罚 `min(0, nvy - (-safe_vy))` → 当 nvy < -safe_vy 时惩罚，避免影响正常飞行和着陆的轻微下降。
   - 系数校准：progress per-step ≈ 12.45/101.6 ≈ 0.122。目标惩罚 per-step ≤ 0.3×主信号 ≈ 0.037。设 w_desc = 0.5，若 nvy = -0.8（快速下坠），惩罚 ≈ -0.5×(0.8-0.3) = -0.25，约 2× progress per-step——在 episode 末尾会迅速抵消 progress。在正常飞行 vy≈0 时惩罚为 0。

2. **减少 lateral_drift_penalty 系数**：从 0.2 降到 0.1，避免过度限制水平机动能力，同时保留对剧烈水平漂移的约束。

**保持不变**：progress（职责正确但需要 counterbalance）、stability_penalty（形态正确）、landing_bonus（结构正确，gate 后续调整）。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observation indices (assumed 2D lander, similar to LunarLander‑v2)
    # 0: x position, 1: y position, 2: vx, 3: vy, 4: angle, 5: angular velocity
    # 6: left leg contact, 7: right leg contact (0.0 or 1.0)
    # Target landing pad is assumed at (0.0, 0.0)

    # Extract current and next states
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle, n_ang_vel = next_obs[4], next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # ---------- 1. Main progress: distance to target decreasing ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new              # + when approaching target

    w_progress = 10.0

    # ---------- 2. Stability constraint: attitude & angular velocity ----------
    # Quadratic penalty on tilt and rotation
    w_angle = 0.5
    w_angvel = 0.1
    stability = -w_angle * (n_angle ** 2) - w_angvel * (n_ang_vel ** 2)

    # ---------- 3. Lateral drift constraint: horizontal speed ----------
    w_lat = 0.1  # reduced from 0.2 to avoid over‑restricting horizontal maneuvering
    lateral_drift = -w_lat * (nvx ** 2)

    # ---------- 4. Soft landing bonus (joint‑condition proxy) ----------
    landing_bonus = 0.0
    # Conditions: both legs on ground, nearly upright, gentle vertical speed
    if n_left > 0.5 and n_right > 0.5 and abs(n_angle) < 0.1 and abs(nvy) < 0.2:
        landing_bonus = 10.0

    # ---------- 5. Descending speed penalty (hinge) ----------
    # Penalize fast downward motion that exploits progress reward
    safe_vy = -0.3   # gentle descent allowed; hinge threshold at 60‑80% of termination boundary
    w_desc = 0.5     # tuned so per‑step penalty ≤ 0.3 × progress per‑step
    descending_penalty = w_desc * min(0.0, nvy - safe_vy)  # negative when nvy < -0.3

    # Combine
    total_reward = (w_progress * progress
                    + stability
                    + lateral_drift
                    + landing_bonus
                    + descending_penalty)

    components = {
        "progress": w_progress * progress,
        "stability_penalty": stability,
        "lateral_drift_penalty": lateral_drift,
        "landing_bonus": landing_bonus,
        "descending_penalty": descending_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号齐全但校准错误——progress 对"接近目标"的奖励被快速下坠 exploit，缺乏对下降速度的约束。
- **behavior**: agent 快速坠毁（len≈100, terminated=100%, 80% 在 150 步内 score<-50）。
- **signal**: progress 信号过强且有方向盲区，无下降速度惩罚，landing_bonus 无法引导 agent 发现安全降落路径。
- **level**: Level 1
- **hypothesis**: 加入下降速度 hinge 惩罚后，agent 将被迫学习控制下降速度，延长存活时间，增加发现 landing_bonus 触发区域的机会。
- **risk**: 惩罚过强可能导致 agent 悬停而不下降，需监控 descending_penalty 的 active_rate 和 episode length。若 len 从 101→200+ 但 score 仍未提升，则下降速度惩罚过强，需降系数。