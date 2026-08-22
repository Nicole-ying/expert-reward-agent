# Response Record

---

## 1. 诊断报告

### evidence

**全局面貌（read_memory + read_training_feedback）：**

| iter | score | term | len | 核心特征 |
|------|-------|------|-----|---------|
| 1 | -159 | T20 | 303 | contact_reward 死组件 |
| 2 | -36 | S20 | 1000 | landing_proximity 99% 主导→盘旋 |
| 3 | -435 | T1/S19 | 967 | landing_proximity 改改善量→崩溃 |
| 4 | -59 | T1/S19 | 978 | landing_proximity 99.5% 主导→盘旋 |
| 5 | -124 | T20 | 68 | 加 fuel_penalty，着陆过快 |
| 6 | -119 | T20 | 68 | approach_reward 86% 主导，全部负贡献 |
| **7** | **252** | **T20** | **308** | **首次正分，架构突破** |

**iter_6→iter_7 的关键变化**（read_reward_code 对比）：
- iter_6: `approach_reward`（实质=progress，w=2.0）→ ep_sum=-66.67，占 magnitude 86%，被 exploit
- iter_7: 删除 landing_proximity，改回 progress_reward (w=2.0) + 软门控 vel_penalty/orientation_penalty/fuel_penalty → 三项平衡，全部健康

**本轮（iter_7）组件状态**：

| 组件 | ep_sum | active_rate | mag_share | 判定 |
|------|--------|-------------|-----------|------|
| progress_reward | +2.67 | 95.2% | 33.1% | ✅ 健康 |
| vel_penalty | -2.52 | 95.7% | 30.5% | ✅ 健康 |
| fuel_penalty | -2.80 | 51.6% | 33.9% | ✅ 健康 |
| **orientation_penalty** | **-0.21** | **100.0%** | **2.5%** | ⚠️ 全覆盖惩罚 |

**Checkpoint 轨迹**：cp1=-151.7 → cp8=247.6 → cp9=197.7（回落）→ cp10=262.8（恢复）。整体稳步上升，cp9 有小幅震荡，但最终恢复。

**历史上 landing_proximity_reward 的 exploit 记录**（get_component_history）：
- iter_2: magnitude_share=99%, ep_sum=+2704 — 盘旋不降
- iter_4: magnitude_share=99.5%, ep_sum=+4731 — 再次盘旋

→ 任何"靠近即奖"的状态值奖励在此环境中极易被 exploit，iter_7 移除它是正确的。

### behavior_diagnosis

策略当前行为：**所有 episode 成功着陆（20/20），但着陆质量有明显差异**（score range 49.7~291.2，6x 差距）。核心问题不在于"能否着陆"，而在于"着陆质量的一致性"。

当前软门控 `gate = 1/(1+3d)` 的问题：
- 在 d=3.0 时 gate=0.1，仍提供不可忽略的 penalty → orientation_penalty 远距离也在激活
- 在 d=0→1 的关键着陆区间，gate 从 1.0 降到 0.25，梯度偏弱
- gate 永不为零 → orientation_penalty active_rate=100% 但 magnitude_share 仅 2.5%，本质是噪声

### signal_completeness

当前四个组件覆盖了全部必要职责：
- `goal_proximity` → progress_reward ✅（delta 距离改善量）
- `velocity_damping_near_pad` → vel_penalty ✅（门控速度惩罚）
- `safe_orientation` → orientation_penalty ⚠️（信号太弱，被淹没）
- `action_efficiency` → fuel_penalty ✅（离散费率）

缺失：无。但 orientation_penalty 的引导力严重不足——agent 在整个 episode 中每步都收到角度惩罚，但量级太小（每步均值仅 -0.00068），无法有效区分"姿态良好"和"姿态差"的状态，形同虚设。

### selected_level

**L2** — 改一个组件的数学形态。虽然分数已超目标（252 > 200，超出 26%），但 orientation_penalty 的结构性问题（100% 激活但近乎零贡献）无法通过调系数解决——无论怎么调权重，只要 gate 远距离不为零，它就会持续"全覆盖"且信号稀释。

选择 L2 而非 L1 的核心理由：这不是力度问题，是门控函数的形态问题。当前的 rational gate `1/(1+3d)` 衰减太慢，导致远距离仍有不可忽略的 penalty，稀释了近 pad 区域真正需要的梯度。

### selected_intervention

**目标组件**：软门控 `gate`（被 vel_penalty 和 orientation_penalty 共享）。

**具体修改**：将门控函数从有理函数 `1/(1+3d)` 替换为指数衰减 `exp(-1.5*d)`：

```
旧: gate = 1.0 / (1.0 + 3.0 * distance)
新: gate = 2.718281828 ** (-1.5 * distance)
```

**系数保持不变**：vel_penalty w=-0.3，orientation_penalty w=-0.2，progress_reward w=2.0，fuel_penalty 不变。

**为什么是 -1.5？** 对比新旧 gate 在关键距离的值：

| distance | 旧 gate | 新 gate | 效果 |
|----------|---------|---------|------|
| 0.0 | 1.000 | 1.000 | 相同 |
| 0.5 | 0.400 | 0.472 | 略强（+18%），着陆关键区信号增强 |
| 1.0 | 0.250 | 0.223 | 略弱（-11%），过渡区基本持平 |
| 2.0 | 0.143 | 0.050 | 大幅减弱（-65%），远距离信号快速归零 |
| 3.0 | 0.100 | 0.011 | 基本归零，orientation_penalty 在远距离实质失活 |

### falsifiable_hypothesis

指数门控比有理门控更适合着陆任务，因为：
1. **近 pad 区梯度更陡**：在 d=0 处，指数 gate 导数=-1.5，有理 gate 导数=-3.0（更陡但仅限 d≈0 极小邻域）；在 d=0.3~1.0 区间，指数 gate 导数持续优于有理 gate，提供更均匀的"减速-调姿"引导
2. **远距离信号干净归零**：d>2.5 时 gate<0.023，orientation_penalty 实质上只在近 pad 区有贡献，解决"全覆盖惩罚"
3. **信号分离更清晰**：远距离时 progress_reward 主导（鼓励快速接近），近距离时 vel_penalty+orientation_penalty 主导（鼓励减速调姿），两个阶段不再互相干扰

### expected_next_round

- **orientation_penalty 的 active_rate**：应从 100% 降至 ~60-80%（仅在近 pad 区域激活）
- **orientation_penalty 的 magnitude_share**：应从 2.5% 小幅上升至 5-8%（在关键区域更集中，而非被稀释）
- **vel_penalty 的 magnitude_share**：可能小幅下降（远距离不再有微弱惩罚），近 pad 区影响基本持平
- **score**：预期小幅提升或持平（250-280），着陆质量方差应减小
- **终止率**：应保持 20/20

### main_risk

指数门控在 d=2.0~3.0 区间 gate 值比旧 gate 低 3-10x，速度惩罚在此区间几乎消失。agent 可能在接近 pad 的中距离（d≈2-3）发展出过高的速度，到达 d≈1 时来不及减速。缓解因素：progress_reward 仍提供接近动力，且 exp(-1.5) 衰减足够平滑，不会产生突变。如果下一轮出现速度过高的问题，可通过增大 decay rate（如 -2.0）或提高 vel_penalty 系数来修正。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_prev, y_prev = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 1. Progress reward (delta improvement on distance) ──
    #     Positive when approaching the target pad, negative when retreating.
    #     w=2.0 gives strong approach gradient.
    dist_prev = (x_prev * x_prev + y_prev * y_prev) ** 0.5
    dist_curr = (x * x + y * y) ** 0.5
    progress = dist_prev - dist_curr
    progress_reward = 2.0 * progress

    # ── 2. Soft landing gate (velocity + orientation penalties near pad) ──
    #     Exponential gate: decays faster than the old rational 1/(1+3d),
    #     localizing penalties to the near-pad region where they matter.
    #     At d=3.0, gate≈0.01 (effectively zero) vs old gate≈0.10,
    #     fixing orientation_penalty's 100% active-rate noise issue.
    distance = dist_curr
    gate = 2.718281828 ** (-1.5 * distance)

    vel_penalty = -0.3 * gate * (vx * vx + vy * vy + 0.1 * angular_vel * angular_vel)
    orientation_penalty = -0.2 * gate * (angle * angle)

    # ── 3. Fuel efficiency (light penalty on engine use) ──
    if action == 2:                      # main engine
        fuel_penalty = -0.02
    elif action == 1 or action == 3:     # orientation engines
        fuel_penalty = -0.005
    else:                                # no engine
        fuel_penalty = 0.0

    total_reward = progress_reward + vel_penalty + orientation_penalty + fuel_penalty

    components = {
        'progress_reward': progress_reward,
        'vel_penalty': vel_penalty,
        'orientation_penalty': orientation_penalty,
        'fuel_penalty': fuel_penalty,
    }
    return float(total_reward), components
```
