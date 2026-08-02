# 设计理由
本轮采用 **Level 2 结构变换**，将上一轮的独立高度惩罚（`height_penalty`）替换为连续高度安全门控（`height_gate`），并乘入前进奖励。  
- 上一轮 `height_penalty` 的 active_rate 仅 15.8%，说明多数步骤高度处于安全区间，但机器人仍迅速终止（len=13），表明独立惩罚无法在倾倒前提供足够强的梯度引导。
- 新设计把高度信号变成一个 **乘法门控**：当身体高度接近终止边界（≤0.2 或 ≥1.0）时，门控值平滑衰减至 0，从而抑制前进奖励；在安全中心区域（≈[0.3, 0.9]）保持门控=1，不对前进奖励造成折扣。  
- 这使高度越界的“惩罚”表现为对前进奖励的打折，而非额外增加负向惩罚，避免总奖励被大量独立惩罚项主导，让 agent 在探索阶段更容易存活并学习行走。

**系数校准**：
- 门控斜坡宽度设为 0.1 米（从终止边界 0.2 到安全值 0.3，及从 1.0 到 0.9），确保在“即将终止”的风险区域提供平滑、非零的梯度。
- `lateral_penalty` 和 `upright_penalty` 维持原系数，暂不调整，以便本轮观察单一组件的改进效果。后续若仍存活但负分，可继续消减这些辅助惩罚。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next_obs
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]

    # Body uprightness (1.0 = perfectly upright, 0.0 = tilted)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    # Guard against tiny numerical overshoot
    body_up_z = max(0.0, min(1.0, body_up_z))

    # ---------- Forward progress (bounded, only positive velocity) ----------
    vx = max(0.0, body_x_vel)
    forward_reward = vx / (1.0 + vx)          # bounded in [0, 1)

    # ---------- Height safety gate (replaces independent height penalty) ----------
    # Safe zone: [0.3, 0.9]; termination boundaries: 0.2 and 1.0
    # Ramp width 0.1 on each side, giving linear decay to 0 at the boundaries
    low_gate  = max(0.0, min(1.0, (body_z - 0.2) / 0.1))   # 0 at 0.2, 1 at 0.3
    high_gate = max(0.0, min(1.0, (1.0 - body_z) / 0.1))   # 0 at 1.0, 1 at 0.9
    height_gate = low_gate * high_gate                     # 1 inside safe zone, decays near boundaries

    # ---------- Upright posture penalty ----------
    upright_penalty = (1.0 - body_up_z)**2

    # ---------- Lateral slip penalty ----------
    lateral_penalty = abs(body_y_vel)

    # Weights
    w_forward  = 1.0
    w_upright  = 5.0
    w_lateral  = 0.5

    total_reward = (w_forward * height_gate * forward_reward
                    - w_upright  * upright_penalty
                    - w_lateral  * lateral_penalty)

    components = {
        "gated_forward":   w_forward * height_gate * forward_reward,
        "upright_penalty": w_upright * upright_penalty,
        "lateral_penalty": w_lateral * lateral_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少高度门控/提前预警信号；惩罚过重导致探索梯度被淹没，agent 无法生存。
- **behavior**: 平均 13 步即高度越界终止，策略尚未习得任何有效步态。
- **signal**: lateral_penalty 和 upright_penalty 占比过高，独立高度惩罚无法在早期提供纠正引导。
- **level**: Level 2（结构变换，独立惩罚 → 乘法门控）
- **hypothesis**: 将高度约束转化为前进奖励的软化阀门，能保留正向梯度并允许更长时间的探索，从而使策略学会维持高度并最终产生前进。
- **risk**: lateral_penalty 与 upright_penalty 系数仍可能偏高，若生存长度回升但总分仍为负，需后续降低这些惩罚项的权重。