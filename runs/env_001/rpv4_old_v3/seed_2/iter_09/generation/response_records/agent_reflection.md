# Response Record

现在我已经有了完整的图景。让我整理分析并输出。

---

## 诊断报告

### 1. evidence

| 来源 | 关键数据 |
|---|---|
| `read_memory()` | iter_8 骨架大改（L3）首次拿到正分 26.45；iter_9 用同样代码再训，得分完全不变 26.45 |
| `read_training_feedback(9)` | terminated 20/20，len=206.5；progress +2.67，vel_damping −1.58，fuel_penalty −1.23，angle_stab −0.26 |
| Checkpoint | **cp40%=0.00, cp80%=0.00 — 训练全程零改善** |
| `get_component_history("progress")` | ⚠️ 刚激活，仅 L1 调参 |
| `get_component_history("vel_damping")` | active 99.9%，全覆盖惩罚（−27% signed share） |
| `get_component_history("angle_stab")` | active 100%（−4.5%） |
| `get_component_history("fuel_penalty")` | active 59.6%（−21%） |
| iter_5 reflection | success_landing_bonus 状态值版 → hovering（len=740, bonus=1195/ep） |
| iter_6 reflection | 改为 transition-based + 收紧阈值 → **直接死亡**（bonus active=0%） |
| iter_7 reflection | 再次 transition-based + 宽松阈值 → 仍接近死亡（active=0.1%） |

### 2. behavior_diagnosis

策略当前行为：agent 学会了朝目标 (0,0) 移动以获取 progress 奖励，但**从未学会减速和软着陆**。由于没有终端成功信号，agent 对"撞地"和"软着陆"一视同仁——只要接近目标就有 progress 正奖励。20/20 terminated 大概率都是高速碰撞终止，只是恰好位置接近目标所以总分勉强为正。Checkpoint 全程平坦（0→0→0）证明 reward 没有提供可学习的梯度：无论 agent 做什么，每步 reward 几乎相同。

### 3. signal_completeness

| 语义功能 | 状态 | 说明 |
|---|---|---|
| 接近/进度信号 | ✅ progress | 每步距离改善量，弱但存在 |
| 终端成功信号 | ❌ **缺失** | iter_8 L3 重构时被移除，这是致命缺口 |
| 终端失败信号 | ❌ 缺失 | 无 crash penalty |
| 稳定性约束 | ✅ vel_damping + angle_stab | 全覆盖但可控 |
| 效率约束 | ✅ fuel_penalty | 中等覆盖率 |

**缺口判断**：终端成功信号是刚性缺失。没有它，agent 永远无法区分"撞地"和"着陆"，目标分数 200 不可能达成。

### 4. selected_level

**L2** — 添加一个新组件 `success_landing_bonus`，引入 transition-based 数学形态。

触发条件：当前骨架缺少终端信号，而历史所有迭代（iter_1–7）都携带某种形式的 success 信号。这不是换框架（L3），而是补回一个被 iter_8 L3 误删的关键组件。progress 被标记 L1-only，不改它。

### 5. selected_intervention

**添加 `success_landing_bonus`**：

- **触发方式**：transition-based（从「无双脚着地」到「双脚着地」的瞬间），避免 iter_5 的 per-step 滞留劫持
- **阈值**：dist < 0.4，speed < 1.0，|angle| < 0.5 —— 比 iter_6（全 0.3，太紧）宽松，比 iter_5 的无限重复版安全
- **奖金**：+100.0，一次性的。若 20 集全成功，贡献 +100，配合 progress~+3 可将总分推至 ~100+；即使只有一半成功，也能提供明确的梯度信号
- 其余组件（progress, vel_damping, angle_stab, fuel_penalty）**完全不动**

### 6. falsifiable_hypothesis

> 引入 transition-based 成功奖金后，agent 将首次获得「软着陆 vs 撞地」的可区分梯度。因为奖金只在接触瞬间触发一次（不可重复收集），agent 无法通过 hovering 劫持；阈值足够宽松（dist<0.4, speed<1.0）使 agent 在当前 len≈206 的行为基础上即可偶然触发，从而让 PPO 看到正向信号并沿着梯度改进。

### 7. expected_next_round

| 指标 | 预期变化 | 原因 |
|---|---|---|
| score | ↑ 30–80 | 成功奖金直接贡献，且引导减速 |
| success_landing_bonus active_rate | 0.5%–5% | 初期稀疏但非零（阈值宽松） |
| success_landing_bonus ep_sum | 5–50 | 部分 episode 触发 |
| len | ~150–250 | 不应暴涨（transition-based 防滞留） |
| progress ep_sum | ~持平或略降 | 减速可能减少激进接近 |
| checkpoint | 应出现上升趋势 | 首次出现可学习梯度 |

### 8. main_risk

**风险**：即使 transition-based，agent 可能学会"弹跳 exploit"——着陆后弹起再着陆，反复触发 bonus。若出现此情况（len↑↑、bonus ep_sum↑↑），下一轮应将阈值收紧至 dist<0.3, speed<0.6 或加入「每人仅一次」计数器逻辑。但当前更紧迫的风险是**阈值太紧导致再次死亡**（iter_6/7 的前车之鉴），因此优先保证信号可达。

---

## 新奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar lander style 2D flyer landing.
    Main goal: reach target platform (x≈0, y≈0) with low velocity, upright.
    Components:
        A: progress (distance reduction towards target)  -- main learning signal
        B: velocity damping (next_obs squared velocity)  -- stability constraint
        C: orientation stabilization (next_obs angle/angular velocity) -- stability constraint
        D: fuel penalty (per action cost)                -- efficiency / mild action cost
        E: success_landing_bonus (transition-based)      -- terminal success signal, one-time
    """
    # Extract states
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    
    # Current and next distance to target (Euclidean)
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    
    # Component A: progress – positive when moving closer to (0,0)
    progress = dist - next_dist  # positive if approaching
    w_progress = 2.0
    component_A = w_progress * progress

    # Component B: velocity damping – penalise large speeds in next step
    vx_n, vy_n = next_obs[2], next_obs[3]
    vel_penalty = vx_n**2 + vy_n**2
    w_vel = 0.05
    component_B = -w_vel * vel_penalty

    # Component C: orientation stabilization – penalise tilt and rotation
    angle_n, angvel_n = next_obs[4], next_obs[5]
    ang_penalty = angle_n**2 + angvel_n**2
    w_ang = 0.05
    component_C = -w_ang * ang_penalty

    # Component D: fuel penalty – light cost for any engine use
    fuel_cost = 0.0 if action == 0 else 1.0
    w_fuel = 0.01
    component_D = -w_fuel * fuel_cost

    # Component E: success landing bonus – one-time bonus on foot contact transition
    # Trigger: both support legs transition from OFF to ON in a single step
    # AND soft-landing conditions are met (moderate thresholds to avoid dead component)
    left_contact = obs[6]
    right_contact = obs[7]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    prev_both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    curr_both_contact = (n_left_contact > 0.5) and (n_right_contact > 0.5)
    contact_made = curr_both_contact and (not prev_both_contact)

    # Soft landing conditions: near target, slow, upright
    landing_dist = (nx**2 + ny**2) ** 0.5
    landing_speed = (vx_n**2 + vy_n**2) ** 0.5
    landing_tilt = abs(angle_n)

    is_soft_landing = (landing_dist < 0.4) and (landing_speed < 1.0) and (landing_tilt < 0.5)

    w_success = 100.0
    component_E = w_success if (contact_made and is_soft_landing) else 0.0

    total_reward = component_A + component_B + component_C + component_D + component_E

    components = {
        "progress": component_A,
        "vel_damping": component_B,
        "angle_stab": component_C,
        "fuel_penalty": component_D,
        "success_landing_bonus": component_E
    }

    return float(total_reward), components
```
