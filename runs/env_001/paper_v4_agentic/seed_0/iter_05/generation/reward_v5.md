# 设计理由

## 行为诊断
第 4 轮从第 3 轮的 `landing_contact_bonus`（连续接触 × 距离门控）切换到稀疏的四因子乘积 `precise_landing_bonus`，导致 agent 在绝大多数状态下得不到任何正向奖励信号。训练崩溃：episode length 从 1000 断崖暴跌到 68，所有 episode 以 crash/出界等失败模式终止，得分断崖式下降。  
组件统计证实：`precise_landing_bonus` 的 active_rate 仅 0.5%（几乎不触发），其正贡献无法弥补惩罚，使得总 reward 分布明显偏负，无法维持存活行为。

## 干预点
本轮只能修改一个组件，优先处理主因——正向着陆奖励从可用信号塌缩为无用信号。  
**恢复类似第 3 轮的 `landing_contact_bonus`**（连续接触因子 × 距离门控），避免因乘积过严导致稀疏；同时 **同步降低 `landing_safety_penalty` 的系数**，使其满足惩罚 per‑step ≤ 主信号 per‑step 的 0.5× 的设计约束，防止惩罚过度压制 progress。

## 结构变换（Level 2）
1. **移除** `precise_landing_bonus`（四因子乘积）。
2. **新增** `landing_contact_bonus`：
   - 连续接触因子：`contact = (n_left + n_right) / 2.0`
   - 距离门控：`gate = 1.0 / (1.0 + 3.0 * dist_next)`，保证远处几乎无奖励，不干扰早期探索。
   - 奖励：`2.0 * contact * gate`（总上限 2.0，远处衰减极快）。
3. **调整** `landing_safety_penalty` 的系数：
   - `vel_pen` 系数 `0.1 → 0.03`，`ang_pen` 系数 `0.05 → 0.02`，`tilt_pen` 系数 `0.1 → 0.03`。
   - 原 per‑step 惩罚 ≈ 0.017 > progress per‑step ≈ 0.016，降低至约 0.005，符合 ≤ 0.5× 主信号的设计上限。
4. 删除僵尸组件 `x_boundary_penalty`（active_rate=0%），简化奖励。

## 系数校准
- 主信号 per‑step（progress + landing_contact_bonus）预估在 0.015–0.03 范围。
- 惩罚 per‑step 约 0.005，远小于主信号的 0.3×。
- `landing_contact_bonus` 的最大值 2.0 不会超过主信号 2×。
- 门控参数 3.0 使 gate 在 `dist=0.5` 时约 0.4，`dist=1.0` 时约 0.25，保持渐近引导。

## 预期效果
- Agent 重新获得接近目标并接触地面的连续正向激励，能够持续存活并向目标移动。
- 降低的安全惩罚允许探索，同时仍然限制接近目标时的激进机动。
- Episode length 回升，score 恢复并向 150+ 推进。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation variables
    x, y = obs[0], obs[1]
    x_v, y_v = obs[2], obs[3]
    angle = obs[4]
    ang_v = obs[5]

    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ---------- 1. Progress reward: moving toward the landing pad (0,0) ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next
    progress_reward = 1.0 * progress

    # ---------- 2. Landing contact bonus (continuous, gated by distance) ----------
    contact = (n_left + n_right) / 2.0
    gate_contact = 1.0 / (1.0 + 3.0 * dist_next)
    landing_contact_bonus = 2.0 * contact * gate_contact

    # ---------- 3. Landing softness / safety penalty (reduced coefficients) ----------
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)

    gate_safety = 1.0 / (1.0 + 5.0 * dist_next)
    landing_safety_penalty = (0.03 * vel_pen + 0.02 * ang_pen + 0.03 * tilt_pen) * gate_safety

    # ---------- Total reward ----------
    total_reward = progress_reward + landing_contact_bonus - landing_safety_penalty

    components = {
        "progress_reward": float(progress_reward),
        "landing_contact_bonus": float(landing_contact_bonus),
        "landing_safety_penalty": float(landing_safety_penalty)
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测均已使用，信号覆盖齐全；问题在于主正向奖励（precise_landing_bonus）因四因子乘积过严而塌缩为 0，导致 agent 无学习信号。
- **behavior**: 快速 crash/出界，episode length 从 1000 暴跌至 68，无法存活。
- **signal**: 缺失连续且不易塌缩的接近着陆正向奖励；安全惩罚过重。
- **level**: Level 2（单组件结构替换 + 系数校准）
- **hypothesis**: 恢复连续接触 × 距离门控的正向奖励，同时将安全惩罚降至主信号的 0.5× 以下，agent 将重新获得向目标移动并存活的能力，分数恢复至第 3 轮水平。
- **risk**: 若安全惩罚过度削弱，agent 可能在接近目标时仍保持较高速度导致 crash；需后续根据存活长度和最终速度进一步微调。