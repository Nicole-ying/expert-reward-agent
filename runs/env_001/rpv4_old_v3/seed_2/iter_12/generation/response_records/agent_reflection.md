# Response Record

Now I have all the data I need. Let me compile the diagnosis and finalize the intervention.

---

## 1. 诊断报告

### evidence

| 来源 | 关键数据 |
|---|---|
| `read_training_feedback(11)` | score=-7.59, len=1000, terminated=0/20, truncated=20/20. `landing_quality`: ep_sum=0.00, active_rate=**0.0%** — 完全死组件 |
| `read_training_feedback(11)` cp | cp40=-33.07 → cp80=-6.78, landing_quality 仅在 cp80 闪现 ep_sum=4.348 (active=0.2%), 最终策略归零 |
| `read_memory()` | iter_9 (sparse bonus w=100): score=**145.17**, term=9/20; iter_10 (sparse bonus w=100): score=131.35, term=11/20; iter_11 (continuous landing_quality): score=**-7.59**, term=0/20 |
| `get_component_history("success_landing_bonus")` | iter_9: ep_sum=5330, active=7.4%; iter_10: ep_sum=6100, active=7.3% — 被 exploit (弹跳重复触发 60+次/ep) |
| `get_component_history("landing_quality")` | iter_11: ep_sum=0.00, active=0.0% — 一次性归零 |
| `read_past_reflection(11)` | 上一轮诊断：将稀疏 bonus 替换为连续 per-step `landing_quality`，理由是对抗弹跳 exploit。结果：landing_quality 完全死掉 |
| `read_reward_code(9)` (best) | 使用稀疏 transition-based `success_landing_bonus`, w=100, 软着陆阈值 dist<0.4 / speed<1.0 / tilt<0.5 |

### behavior_diagnosis

策略当前行为：**永不尝试着陆**。agent 学会了盘旋在目标区域附近，通过 progress 信号（dist→next_dist 改善）获取正奖励来对抗 fuel_penalty，但从不使双足同时触地。20 个评估回合全部 timeout 在 1000 步。这是 iter_11 将稀疏 bonus 替换为 `landing_quality` 的直接后果——新组件要求双足同时触地才给奖励，但 agent 从未到达这个状态，因此失去了唯一的着陆引导信号。

历史证明：稀疏 transition-based bonus（iter_9-10）**有效诱导了着陆行为**（9-11/20 成功终止），但被弹跳 exploit（60+ 次触发/ep）。iter_11 的 L2 替换方案过度修正，完全消灭了着陆信号。

### signal_completeness

必要职责覆盖情况：
- `goal_progress_guidance` ✅ — progress 组件正常工作（ep_sum=26.43）
- `soft_landing_safety` ❌ — landing_quality 完全死掉。没有信号告诉 agent「着陆是好的」
- `energy_efficiency` ✅ — fuel_penalty 正常（但 len=1000 时固定 -10 天花板）
- `sparse_mission_success_reward` ❌ — 缺失。iter_9-10 的 success_landing_bonus 承担此职责但被移除

**缺失信号**：着陆成功奖励。这是 agent 不尝试着陆的根本原因。

### selected_level

**L2** — 修改一个组件的数学形态：从 iter_11 的连续 per-step `landing_quality`（需双足触地才激活）回到稀疏 transition-based bonus，但新增 **高度门控**（`y > 0.1`）作为防 exploit 措施。

选择理由：L1（只调系数）无法解决 landing_quality 的结构性缺陷（它要求 agent 已经着陆才能获得奖励）；L3（换框架）不必要，因为 iter_9-10 证明稀疏 bonus 框架可以工作。

### selected_intervention

**目标组件**：`landing_quality` → 替换回 `success_landing_bonus`

**具体修改**：
1. 恢复稀疏 transition-based bonus（`contact_made = curr_both_contact and not prev_both_contact`）
2. 新增 **高度门控**：`was_descending = y > 0.1`（着陆前一步 agent 身体高度必须 > 0.1，即双足未触地状态）
3. **降低权重**：w=50（从 100 降低），减轻 exploit 的量级
4. 保留软着陆条件：dist<0.4, speed<1.0, tilt<0.5

高度门控的防 exploit 原理：首次着陆时 agent 从 y≈1.0+ 下降，y>0.1 必然满足。着陆后 agent 处于 y≈0。弹跳时身体最多上升到 0.1-0.2，重新触地时前一 step 的 y 可能在 0.05-0.12 之间——若 y≤0.1 则门控阻止触发。只有显著的弹跳（身体上升超过 0.1）才可能重新触发，显著减少了 exploit 机会。结合 w=50，即使少量重触发也不会主导 reward。

### falsifiable_hypothesis

恢复稀疏 transition-based 着陆奖励将重新建立「着陆是有价值的」这一梯度信号。agent 将开始尝试使双足触地，终止率应从 0/20 上升。高度门控 + 降低权重应使 ep_sum 控制在 50-250 范围（1-5 次触发），而非 iter_10 的 6100。

### expected_next_round

- `score`: 应从 -7.59 回升至正区间（目标 > 30）
- `terminated`: 应从 0/20 上升至至少 3-5/20
- `success_landing_bonus`: active_rate 应在 0.5-3% 范围，ep_sum 在 50-250（非 exploitative）
- `progress`: ep_sum 可能在 20-26 范围（与 iter_11 持平）
- `fuel_penalty`: ep_sum 绝对值应下降（episode 因成功着陆而提前终止）
- `len`: 应从 1000 下降（部分 episode 成功终止）

### main_risk

**最大风险**：高度门控 `y > 0.1` 可能对极慢着陆（agent 在 y=0.05 处悬停然后触地）过于严格，导致 bonus 仍然死掉。如果下一轮 active_rate 仍然为 0%，需要将门控阈值降至 0.03 或完全移除门控、改用其他防 exploit 手段（如限速阈值收紧）。

---

## 2. 完整代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar lander style 2D flyer landing.
    Main goal: reach target platform (x≈0, y≈0) with low velocity, upright.
    
    Components:
        A: progress (distance reduction towards target)       -- main learning signal
        B: velocity damping (next_obs squared velocity)       -- stability constraint
        C: orientation stabilization (next_obs angle/angvel)  -- stability constraint
        D: fuel penalty (per action cost)                     -- efficiency / mild action cost
        E: success_landing_bonus (transition-based, height-gated) -- terminal success signal
    
    iter_12 change (L2): Replaced dead landing_quality (continuous per-step, required 
    both feet already on ground) with restored sparse transition-based success_landing_bonus.
    Added height gate (y > 0.1) to prevent bounce exploitation — after landing at y≈0, 
    small bounces won't clear the gate. Reduced weight from 100→50 to further limit 
    exploit magnitude. This restores the landing gradient that iter_9-10 proved effective.
    """
    # Extract states
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    
    # Current and next distance to target (Euclidean)
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    
    # Component A: progress – positive when moving closer to (0,0)
    progress = dist - next_dist  # positive if approaching
    w_progress = 20.0
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
    # Trigger: both support legs transition from OFF to ON in a single step,
    # agent was at meaningful height (y > 0.1, i.e. feet off ground),
    # AND soft-landing conditions are met.
    # Height gate prevents bounce exploitation: after landing at y≈0, 
    # only significant bounces (body rising >0.1) can re-trigger.
    left_contact = obs[6]
    right_contact = obs[7]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    prev_both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    curr_both_contact = (n_left_contact > 0.5) and (n_right_contact > 0.5)
    contact_made = curr_both_contact and (not prev_both_contact)

    # Anti-bounce height gate: agent must have been above ground before this step
    was_descending = y > 0.1

    # Soft landing conditions: near target, slow, upright
    landing_dist = (nx**2 + ny**2) ** 0.5
    landing_speed = (vx_n**2 + vy_n**2) ** 0.5
    landing_tilt = abs(angle_n)

    is_soft_landing = (landing_dist < 0.4) and (landing_speed < 1.0) and (landing_tilt < 0.5)

    w_success = 50.0  # reduced from 100 to limit exploitation magnitude
    component_E = w_success if (contact_made and is_soft_landing and was_descending) else 0.0

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
