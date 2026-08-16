# Response Record

---

## 1. 诊断报告

### evidence
- **iter_2 得分**: -36.38（目标 200），terminated=0/20，truncated=20/20——**零成功着陆，全被 1000 步截断**。
- **主导组件**: `landing_proximity_reward` magnitude_share=99%，ep_sum=2704，active_rate=100%。相比之下 `vel_damping` 仅 -26.8（1%），`progress_reward` 仅 +1.04（≈0%），`orientation_penalty` 仅 -0.30（≈0%）。
- **Checkpoint 趋势**: score 从 cp10%=-974 快速升至 cp40%=-114，之后在 -34~-146 之间震荡。`landing_proximity_reward` 在 cp40% 从 520 跳升至 2033——agent 在此阶段发现了 **hovering 策略**。
- **行为事实**: agent 在离目标垫 ~1 单位距离内以近零速度、竖直姿态悬停，每步获取 ~2.7 的 Gaussian 状态奖励，持续 1000 步直至截断。从不触发 settled 终止条件。
- **iter_1 → iter_2 变化**: 替换死组件 `contact_reward`（要求双腿同时接触，太严格，0% 激活）→ `landing_proximity_reward`（连续高斯乘积，100% 激活）。len 从 303→1000（从 crash 变为 hovering），terminated 从 20→0。

### behavior_diagnosis
**Hovering 利用（hovering exploit）**：agent 学会了悬停在目标垫正上方（近距离、低速、竖直），最大化 `landing_proximity_reward` 的 per-step 回报。由于该奖励是纯状态型（state-based），agent 没有动机完成最后几厘米的下降——着陆会导致 episode 终止、奖励流中断。`progress_reward` 的 ep_sum 仅 1.04（每步 ~0.001），证实 agent 在水平面上几乎不做位移，处于稳态悬停。

### signal_completeness
对照环境卡片的 `reward_role_decomposition`：
| 职责 | 当前状态 |
|---|---|
| `goal_proximity` | ✅ 有（landing_proximity_reward + progress_reward），但被 exploit |
| `soft_landing_success` | ❌ **缺失**——无任何着陆成功信号，无 contact 激励 |
| `safe_orientation` | ✅ 有（orientation_penalty），但被淹没（magnitude_share≈0%） |
| `velocity_damping_near_pad` | ✅ 有（vel_damping），激活但不敌 hovering 收益 |
| `action_efficiency` | ❌ 缺失——无燃料/动作惩罚 |

最关键的缺口是 **着陆接触激励**。当前没有任何奖励组件直接奖励支撑腿接触地面。`landing_proximity_reward` 奖励"接近+低速+竖直"，而这个最优状态恰好可以在不接触地面的情况下达成（悬停于 y≈0.1~0.2）。

### selected_level
**L2** — 修改 `landing_proximity_reward` 的数学形态：从**纯状态型 Gaussian 乘积**改为**进度加权型（progress-weighted）**。这不是简单的系数调整（L1），也不是换框架（L3），而是改变同一组件的引导逻辑。

### selected_intervention
**目标组件**: `landing_proximity_reward`

**具体修改**: 将公式从：
```
reward = w * exp(-dist²/σ_d²) * exp(-speed²/σ_s²) * exp(-angle²/σ_a²)
```
改为：
```
reward = w * max(0, dist_prev - dist_curr) * exp(-speed²/σ_s²) * exp(-angle²/σ_a²)
```

核心变化：用 `max(0, progress)` 替换 `exp(-dist²/σ_d²)`。奖励不再取决于"有多近"，而取决于"是否在靠近"。悬停时 progress≈0 → reward≈0，从根本上消除 hovering exploit。速度因子和角度因子保留，确保 agent 以受控方式接近（低速、竖直），而非高速俯冲。

### falsifiable_hypothesis
**如果悬停是 agent 在 iter_2 中的最优策略，那么移除状态型奖励、改为进度型奖励后，悬停将产生零额外收益，agent 必须持续靠近目标垫才能获取正向奖励。这迫使 agent 完成下降→接触→着陆的完整链条。** 预期结果：terminated 比例上升（出现成功着陆），len 下降（不再耗满 1000 步），`landing_proximity_reward` 的 ep_sum 大幅下降（从 2704 → 预计 20~50），magnitude_share 回归健康范围。

### expected_next_round
- `score`: 应显著改善（预计 > -20），因为 agent 不再浪费 1000 步悬停
- `terminated > 0`: 至少出现若干次成功着陆
- `len < 1000`: episode 提前终止
- `landing_proximity_reward` ep_sum 大幅下降（从 2704 → ~20-50），magnitude_share 不再垄断
- `progress_reward` ep_sum 可能上升（agent 需要持续移动）
- `vel_damping` 仍活跃，但因 episode 缩短，累计惩罚减少

### main_risk
**进度型奖励可能过于稀疏**：`max(0, progress)` 每步最大约 0.01~0.1（取决于步长），即使 w=5.0，每步也仅 ~0.05~0.5。如果信号太弱，agent 可能退化为随机探索，得分倒退。缓解措施：w_landing 设为 5.0（原为 3.0），且 `progress_reward`（w=1.0）提供额外的进度信号作为补充。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v3: landing_proximity_reward changed from state-Gaussian to progress-weighted —
        rewards only active approach (dist_prev > dist_curr), eliminating hovering exploit.
    """
    # Extract relevant signals
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # 1. Progress reward (improvement_delta on Euclidean distance)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # positive when approaching
    w_progress = 1.0
    progress_reward = w_progress * progress

    # 2. Orientation penalty (quadratic_penalty)
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Landing proximity reward — progress-weighted (NOT state-based)
    #    Only rewards active approach toward the pad, gated by speed and angle quality.
    #    Hovering (progress≈0) yields zero reward, fixing the iter_2 hovering exploit.
    approach = max(0.0, progress)                        # zero when hovering or retreating
    speed = (vx ** 2 + vy ** 2) ** 0.5
    sigma_speed = 1.0
    sigma_angle = 0.3

    speed_factor = 2.718281828 ** (- (speed ** 2) / (sigma_speed ** 2))
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))

    w_landing = 5.0
    landing_proximity_reward = w_landing * approach * speed_factor * angle_factor

    # 4. Near-pad velocity damping (hinge / gated penalty)
    dist_thresh = 1.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 at far, 1 at origin
    w_vel_damp = 0.5
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    total_reward = progress_reward + orientation_penalty + landing_proximity_reward + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_proximity_reward": landing_proximity_reward,
        "vel_damping": vel_damping
    }
    return float(total_reward), components
```
