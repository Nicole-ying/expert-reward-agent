# Response Record

---

## 诊断报告

### evidence

| 来源 | 关键数据 |
|---|---|
| `read_training_feedback(10)` | score=131.35, success_landing_bonus: ep_sum=6100, magnitude_share=**99.3%**, active_rate=7.3% |
| 同上 | cp40%=0.00, cp80%=0.00 — 训练前80%完全无改善 |
| `read_training_feedback(9)` | success_landing_bonus: ep_sum=5330, share=**99.8%** — 同样的主导模式 |
| `read_training_feedback(8)` | 无 success bonus: progress share=47.5%, vel_damping=27%, 所有信号有意义的份额 |
| `get_component_history("success_landing_bonus")` | iter_09: 5330→iter_10: 6100, 两次均>99% share |
| `get_component_history("progress")` | iter_08: share=47.5% → iter_09: 0.1% → iter_10: 0.5%（w=20仍被淹没） |
| `read_memory()` | best=145.17(iter_9), iter_10回退到131.35(-13.82) |
| `read_past_reflection(10)` | 上一轮意图：提升w_progress 2→20对抗success主导 → **失败，score下降** |

**关键数字**：w_success=100，ep_sum_mean=6100 → 平均每episode触发61次bonus。11/20 episodes成功着陆，即每个着陆episode触发~111次。这是**明确的exploit行为**——着陆器在平台上反复弹跳，每次both_contact转换都触发100奖励。

### behavior_diagnosis

策略已学会"到达平台并反复弹跳"来刷取success_landing_bonus。这解释了：
- **ep_sum=6100**（w=100时需61次触发/ep）：着陆器弹跳→双脚离地→再触地→反复触发transition bonus
- **checkpoints=0.00**：训练前80%阶段，策略尚未发现平台，所有dense信号被success bonus淹没，无有效梯度
- **iter_10比iter_9更差**：w_progress的boost（2→20）让策略更激进地冲向平台，弹跳更频繁（fuel_penalty从-4.27恶化到-7.93），但着陆质量下降

核心问题：**稀疏的一次性bonus被exploit，且其巨大magnitude(99.3%)淹没了所有dense shaping信号。**

### signal_completeness

必要职责覆盖情况：
- `goal_progress_guidance` ✅ progress组件 — 但被淹没（share=0.5%）
- `soft_landing_safety` ⚠️ vel_damping + angle_stab — 存在但被淹没（share≈0%）
- `energy_efficiency` ✅ fuel_penalty — 存在但弱
- `sparse_mission_success_reward` ❌ **当前形式被exploit**，需要重构

缺少的是：一个**不会被exploit的着陆奖励信号**，它应该在着陆期间持续提供梯度，而不是在接触瞬间给一大笔。

### selected_level

**L2** — 改变success_landing_bonus的数学形态：从稀疏一次性bonus（transition-triggered）改为**连续着陆质量奖励**（per-step while landed）。这不是调参(L1)，因为信号发放逻辑根本改变。也不是L3，因为保留progress/vel/angle/fuel框架不变。

### selected_intervention

**目标组件**：`success_landing_bonus` → 重命名为 `landing_quality`

**修改内容**：将稀疏transition-based bonus改为连续per-step着陆质量奖励：
- 每当双脚着地（`curr_both_contact`），给予 `w_landing * quality_score` 奖励
- `quality_score = prox_score × speed_score × tilt_score`，每项在0~1之间
- 不要求transition——只要双脚在地上就给，弹跳期间中断
- 完美着陆（dist≈0, speed≈0, tilt≈0）每步得~5分，持续20步≈100分
- 弹跳意味着离地期间零奖励，无exploit动机

同时保留iter_10的 w_progress=20.0（已验证progress可被boost）。

### falsifiable_hypothesis

将稀疏transition bonus改为连续质量奖励后：
1. **magnitude_share应大幅下降**（从99.3%→预计20-40%），因为不再有弹跳刷分
2. **progress/vel_damping/angle_stab的share应回升**，恢复有效梯度
3. **checkpoints不应再为0.00**，因为dense信号在训练全程有效
4. **terminated比例应上升**（更多真正稳定着陆，而非弹跳）

### expected_next_round

- score：应回升至140-170区间（消除exploit后真实着陆能力应接近iter_9水平，配合更强progress）
- landing_quality ep_sum：预计200-500（w=5×20步×0.5~1.0 quality），不再6000+
- progress share：应恢复到5-15%
- 终止率：terminated应≥12/20
- Checkpoint：cp40%和cp80%应出现非零分

### main_risk

着陆质量奖励是per-step的，策略可能学会"着陆后什么也不做等待termination"来收集奖励。但这是**期望行为**（稳定着陆后保持不动正是目标），且termination由物理引擎自动触发（body settles），策略无法无限拖延。风险较低。

另一个风险：如果w_landing=5.0太小，着陆激励不足，策略可能退化为iter_8水平（无着陆动机，全靠progress引导）。若下一轮terminated下降，需提高w_landing。

---

## 完整代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar lander style 2D flyer landing.
    Main goal: reach target platform (x≈0, y≈0) with low velocity, upright.
    
    Components:
        A: progress (distance reduction towards target)       -- main learning signal (STRENGTHENED)
        B: velocity damping (next_obs squared velocity)       -- stability constraint
        C: orientation stabilization (next_obs angle/angvel)  -- stability constraint
        D: fuel penalty (per action cost)                     -- efficiency / mild action cost
        E: landing_quality (per-step reward while landed)     -- continuous soft-landing signal (L2: was sparse bonus)
    
    iter_11 change (L2): Replaced sparse success_landing_bonus (transition-triggered, 
    w=100 one-time) with continuous landing_quality reward (per-step while both feet 
    on ground, scaled by proximity/speed/tilt quality). This prevents exploitation 
    via bouncing (iter_10: 61 triggers/episode at w=100 → ep_sum=6100, 99.3% share)
    and restores gradient to dense shaping signals.
    """
    # Extract states
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    
    # Current and next distance to target (Euclidean)
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    
    # Component A: progress – positive when moving closer to (0,0)
    progress = dist - next_dist  # positive if approaching
    w_progress = 20.0  # iter_10: verified boostable, kept at 20
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

    # Component E: landing quality – per-step reward while both feet are on ground
    # Replaces the sparse transition-based bonus that was exploited via bouncing.
    # Reward is continuous: every step with both feet down earns quality-scaled reward.
    # Quality drops to 0 for bad landings (high speed, tilt, or far from center).
    # Bouncing interrupts reward → no incentive to bounce.
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]
    curr_both_contact = (n_left_contact > 0.5) and (n_right_contact > 0.5)

    # Landing quality metrics (0=perfect, threshold=zero-quality)
    landing_dist = next_dist  # already computed above
    landing_speed = (vx_n**2 + vy_n**2) ** 0.5
    landing_tilt = abs(angle_n)

    # Quality scores: 1.0 at perfect, linear decay to 0 at threshold
    prox_score = max(0.0, 1.0 - landing_dist / 0.4)      # 0 at dist≥0.4
    speed_score = max(0.0, 1.0 - landing_speed / 1.0)    # 0 at speed≥1.0
    tilt_score = max(0.0, 1.0 - landing_tilt / 0.5)      # 0 at tilt≥0.5

    landing_quality = prox_score * speed_score * tilt_score  # 0.0 to 1.0

    w_landing = 5.0  # per-step reward when landed; ~100 total over 20 stable steps
    component_E = w_landing * landing_quality if curr_both_contact else 0.0

    total_reward = component_A + component_B + component_C + component_D + component_E

    components = {
        "progress": component_A,
        "vel_damping": component_B,
        "angle_stab": component_C,
        "fuel_penalty": component_D,
        "landing_quality": component_E
    }

    return float(total_reward), components
```
