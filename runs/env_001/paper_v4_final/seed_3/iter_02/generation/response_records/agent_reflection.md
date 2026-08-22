# Response Record

# 设计理由

## 信号覆盖审计

### 0.1 终止模式分析
所有20个评估episode都是 `terminated=True`，episode平均长度101.6步。在2D飞行器着陆任务中，常见的终止条件可能包括：飞行器角度过大（倾覆）、飞行器飞出视口边界、或高速撞击地面。结合得分高度负（-111.37）且长度较短，推测飞行器可能以较大的角度或速度移动，触发了某些危险终止条件。由于没有leg接触证据（landing_bonus active_rate仅0.3%），飞行器很可能在着陆前就因不当飞行姿态被终止。

### 0.2 观测使用扫描
当前代码使用了全部8个观测维度：
- `obs[0:2]`（位置）：用于progress计算
- `obs[2:4]`（速度）：用于lateral_drift和landing check
- `obs[4:6]`（角度、角速度）：用于stability惩罚和landing check
- `obs[6:8]`（腿接触）：用于landing_bonus判断

所有观测维度已被使用，但问题在于信号形态设计不佳。

### 0.3 信号缺口判断
**信号齐全但校准问题严重**：所有相关观测已被使用，但landing_bonus活跃率仅0.3%（几乎从不触发），提示着陆条件设置过严或飞行器根本达不到该状态。stability和lateral_drift惩罚活跃率接近100%，说明飞行器在整个飞行过程中持续受到惩罚，但惩罚未能有效引导到安全状态。

### 0.4 僵尸组件检查
- `landing_bonus`：active_rate 0.3% → 几乎失效，需要重构为更可达到的形式。

---

## 1. 行为诊断
飞行器得分高度负（-111）、长度短（101步）、所有episode提前终止。结合stability_penalty和lateral_drift_penalty活跃率接近100%但未能阻止终止，说明**飞行器以不稳定姿态飞行并触发危险终止**。progress组件虽然为正（12.45），但被负分完全淹没。

**核心问题**：缺少对即将发生的灾难性失败的预警信号。当前只有全局二次惩罚，没有在接近危险边界时提供强化信号。

**干预目标**：飞行姿态稳定性——当角度或速度接近危险阈值时，给予更强的结构性惩罚，引导飞行器保持安全姿态。

**方向评估**：这是第一轮迭代，没有历史失败记录。当前方向（通过稳定性约束引导安全飞行）合理，但信号形式需要从全局二次惩罚改为**边界感知的hinge/gate形式**，在安全区域内减轻惩罚，在危险边界附近加大力度。

---

## 2. 干预层级：Level 2 — 结构变换

**选择的变换**：将 `stability_penalty` 从全局二次惩罚改为**软门控（soft health gate）**，在角度较小时不影响主奖励，在角度接近危险值时逐步削减progress奖励。

**理由**：
- 全局二次惩罚使飞行器始终处于负反馈中，无法区分"安全区"和"危险区"
- 改用hinge形式的gate：当abs(angle)在安全范围内→gate≈1（不衰减progress）；当接近危险边界→gate迅速衰减，强烈抑制progress
- 这种结构化惩罚比二次惩罚更具信息量：它告诉agent"保持当前角度即可获得全额进度奖励"

**保留原stability作为辅助**：降低权重，作为微调信号防止角度持续增长。

**变换要点**：
- 删除原二次stability惩罚（作为主惩罚），改为gate乘到progress上
- Gate设计：`smooth_hinge at 0.15 rad`（假设终止边界约0.25-0.3 rad）
- 保留弱角速度惩罚作为辅助平滑信号

---

## 设计校准

1. **主信号per-step**：progress per-step ≈ 12.45/101.6 ≈ 0.122
2. **Gate安全阈值**：设0.15 rad（约8.6°），在终止边界（假设0.25-0.3 rad，约14-17°）的60%处
3. **Gate在安全区**：abs(angle)=0.15时gate=0.5（临界点），abs(angle)<0.1时gate>0.8（安全）
4. **新惩罚系数**：角速度惩罚保持极低（0.1），不干扰主信号

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observation indices
    # 0: x position, 1: y position, 2: vx, 3: vy, 4: angle, 5: angular velocity
    # 6: left leg contact, 7: right leg contact (0.0 or 1.0)

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
    progress = dist_old - dist_new

    w_progress = 10.0

    # ---------- 2. Attitude gate: suppress progress when angle is dangerous ----------
    # Smooth hinge: gate ≈ 1.0 when |angle| << 0.15, gate → 0.2 when |angle| >> 0.15
    # Use tanh for smooth transition; 0.15 rad ≈ 8.6° is safety threshold
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (20.0 * (abs_angle - 0.15)) / (1.0 + 2.718281828 ** (20.0 * (abs_angle - 0.15))))

    # ---------- 3. Lateral drift constraint: horizontal speed ----------
    w_lat = 0.2
    lateral_drift = -w_lat * (nvx ** 2)

    # ---------- 4. Angular velocity penalty: small auxiliary smoothing ----------
    w_angvel = 0.1
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. Landing bonus: soft continuous proxy ----------
    # Both legs touching, nearly upright, gentle speeds
    both_legs = min(n_left, n_right)              # 0.0 to 1.0
    vertical_ok = max(0.0, 1.0 - abs(nvy) / 0.3) # 1.0 when vy≈0, 0 when |vy|>=0.3
    attitude_ok = max(0.0, 1.0 - abs_angle / 0.15) # 1.0 when angle≈0, 0 when |angle|>=0.15

    landing_factor = both_legs * vertical_ok * attitude_ok
    landing_bonus = 3.0 * landing_factor          # up to 3.0, smooth

    # Combine: progress is gated by attitude, then penalties and bonus added
    total_reward = (w_progress * progress * angle_gate
                    + lateral_drift
                    + angvel_penalty
                    + landing_bonus)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_drift_penalty": lateral_drift,
        "angvel_penalty": angvel_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测已被使用，但stability信号形式不当（全局二次惩罚无法区分安全/危险区），landing_bonus几乎从不触发（0.3%）
- **behavior**: 飞行器以不稳定姿态飞行，所有episode提前终止，得分高度负
- **signal**: 缺少灾难性失败预警——角度接近危险边界时无结构化惩罚增强
- **level**: Level 2
- **hypothesis**: 将二次稳定性惩罚改为角度门控（gate进步奖励），在危险角度区域强力抑制progress，迫使agent学习保持小角度以获得全额进度奖励；同时将稀疏landing_bonus改为连续因子，提供渐进式着陆引导
- **risk**: Gate可能在训练早期过度抑制progress导致学习缓慢；若角度阈值0.15设置过严，agent可能不敢移动。通过保留0.2的gate底限（即使大角度也有20% progress）缓解此风险。
