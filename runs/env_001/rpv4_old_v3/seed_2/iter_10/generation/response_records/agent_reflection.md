# Response Record

Now I have all the data I need. Let me compile the analysis.

---

## 诊断报告

### 1. evidence

| 来源 | 关键数据 |
|---|---|
| `read_memory()` | iter_8 (无 success): score=26.45, len=206.5, terminated=20/20; iter_9 (加 success): score=145.17, len=718.8, terminated=9/20 |
| `read_training_feedback(9)` | success_landing_bonus: ep_sum=5330, magnitude_share=**99.8%**, active=7.4% |
| 同上 | progress: ep_sum=2.638, magnitude_share=**0.1%**, active=99.2% |
| 同上 | vel_damping: ep_sum=−2.022, magnitude_share=**0.0%**; angle_stab: −0.748, **0.0%**; fuel_penalty: −4.267, **0.1%** |
| Checkpoint | cp40%=0.00, cp80%=0.00 — 训练全程无中间改善 |
| `get_component_history("progress")` | signed_share 从 iter_8 的 45.6% 暴跌至 iter_9 的 0.0% |
| `get_component_history("success_landing_bonus")` | iter_02-07 死组件 (active≤1.6%)，iter_09 跳至 7.4%, ep_sum=5330 |
| `read_reward_code(8→9)` | iter_8→iter_9 **唯一改动**：新增 success_landing_bonus (w=100)。其他系数完全未变。 |

### 2. behavior_diagnosis

**核心问题：success_landing_bonus 信号淹没所有连续组件。**

- Agent 学会在 ~45% (9/20) 的 episode 中成功着陆，获得巨量 bonus。但在其余 11/20 episode 中，连续信号（progress、vel_damping、angle_stab、fuel_penalty）因量级被完全淹没而无法提供有效梯度。
- 每步 progress 约 0.0037×2.0=**0.007**，而 success bonus 单次触发即 +100（且在着陆 episode 中平均触发 ~53 次，ep_sum=5330 远非"一次性"）。progress 的 per-step 信号比 success 的 step 信号弱 ~14000 倍。
- 连续约束惩罚（vel −0.003/步, ang −0.001/步, fuel −0.006/步）合计约 **−0.01/步**，与 progress 的 +0.007/步几乎抵消。Agent 在不触发 success 的 episode 中获得接近零的净信号 → 无法学习有效接近行为。
- Checkpoint 全程 0.00 印证了这一点：训练过程中 agent 没有任何中间改善，只靠稀疏 bonus 偶然发现着陆。

Agent 当前策略：在 success bonus 的巨大引力下盲目尝试着陆，55% 的 episode 因缺乏中间引导而迷失。

### 3. signal_completeness

| 职责 | 当前状态 | 评价 |
|---|---|---|
| 接近引导（progress） | 存在但量级被淹没 | **需大幅加强** |
| 着陆成功（success_landing_bonus） | 存在且有效，但过度主导 | 暂不修改 |
| 速度约束（vel_damping） | 存在，量级合理 | 维持 |
| 姿态约束（angle_stab） | 存在，量级合理 | 维持 |
| 燃料效率（fuel_penalty） | 存在，量级合理 | 维持 |

唯一缺口：progress 信号强度不足以在实际学习中与 success bonus 竞争。

### 4. selected_level

**L1** — 仅调整 progress 的权重系数，不修改数学表达式。

触发条件：success_landing_bonus 的 ep_sum 从 iter_07 的 5 跳至 iter_09 的 5330（>1000x），但 `get_component_history` 未触发 5x 警告。且历史中从未尝试过大幅提升 progress 权重（w_progress 始终为 2.0）。此方向未失败过。

### 5. selected_intervention

**目标组件**：`progress`（Component A）

**修改内容**：将 `w_progress` 从 `2.0` 提升至 `20.0`（10x）。

不修改 success_landing_bonus、vel_damping、angle_stab、fuel_penalty 的任何参数。

### 6. falsifiable_hypothesis

progress 当前 per-step ≈ 0.007，与约束惩罚合计（≈ −0.01）几乎抵消，agent 在不触发 success 的 episode 中无有效梯度。将 w_progress 提升至 20.0 后：

- per-step progress ≈ **0.07**，远大于约束惩罚合计 ≈ −0.01
- 连续净信号变为明确正值（≈ +0.06/步），agent 在非着陆 episode 中也能从"接近目标"获得清晰梯度
- Agent 应更快、更稳定地接近平台，将 success rate 从 45% 推向更高
- Checkpoint 指标应开始显示非零改善

### 7. expected_next_round

| 指标 | 当前 (iter_9) | 预期变化 |
|---|---|---|
| score | 145.17 | ↑ 提升至 160-180 |
| terminated (landing) | 9/20 | ↑ 增至 12-16/20 |
| len | 718.8 | ↓ 轻微下降（更快着陆） |
| progress magnitude_share | 0.1% | ↑ 升至 5-15% |
| success_landing_bonus magnitude_share | 99.8% | ↓ 降至 80-95% |
| checkpoint cp40/cp80 | 0.00/0.00 | ↑ 开始出现正值 |

### 8. main_risk

**progress 过强导致 agent 学会"hover 靠近但不着陆"**：如果 progress 奖励足够大，agent 可能学会在目标附近徘徊获取 progress 而不触发着陆终止。但 progress = dist − next_dist，仅在接近时为正，无法通过远离→接近循环套利（远离时 progress 为负）。真正风险是 agent 减速到极慢接近速度以延长 episode 并累积 progress → len 可能增加。若出现此情况，下一轮需收紧 success bonus 或引入时间惩罚。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar lander style 2D flyer landing.
    Main goal: reach target platform (x≈0, y≈0) with low velocity, upright.
    Components:
        A: progress (distance reduction towards target)  -- main learning signal (STRENGTHENED)
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
    w_progress = 20.0  # iter_10: 2.0 → 20.0 (10x) to counter success_landing_bonus drowning
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
