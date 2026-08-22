# 设计理由
既往八轮迭代揭示了几个稳定规律：
- 线性速度 `horizontal_velocity` 配合平缓倾斜门控 `gate` 是有效的主学习信号骨架（Iter 8 得分 320.8，首次突破目标）。
- 大幅动作惩罚（≥ 0.1）或凸化速度（v²）会严重抑制探索，导致得分崩溃（Iter 3, 7）。
- 独立的稳定性补偿项（bonus/penalty）效果不如门控，往往引入噪声（Iter 6）。

但当前奖励完全依赖速度-门控乘积，未利用观测中丰富的接触信息（`leg1_contact`, `leg2_contact`）。人类步行直觉表明：清晰的双脚交替支撑能提升步态效率与速度稳定性。因此 Level 3 重建的核心思路是：**在已收敛的成功骨架中注入轻量级的步态交替奖励，引导 agent 在保持安全倾斜的同时形成更明确的单腿支撑节奏**。新信号不会惩罚双支撑，只在交替支撑时给予微小正反馈，不会破坏已有平衡。

系数校准：
- `gate` 沿用 Iter 8 的平方反比形式，倾斜 0.5 rad 时衰减至 0.5，安全边际充足。
- `action_penalty` 保持 0.02，per‑step 约占主信号的 11 %，负担合理。
- `gait_bonus` 系数 0.03：交替时刻贡献 ≈ 0.03，平均每步约 0.015，不会支配总奖励。
- `vertical_penalty` 轻量保留，抑制颠簸。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 姿态门控（与迭代 8 相同的安全前兆信号） ----------
    hull_angle_next = next_obs[0]
    gate = 1.0 / (1.0 + (abs(hull_angle_next) / 0.5) ** 2)

    # ---------- 主学习信号：线性前进速度 × 门控 ----------
    horizontal_velocity = next_obs[2]
    velocity_gated = gate * horizontal_velocity

    # ---------- 轻量步态交替奖励 ----------
    # 当双腿接触状态不同（一腿支撑、一腿摆动）时给予正奖励；
    # 双支撑或双离地时奖励为 0。不惩罚，不干扰探索。
    contact_diff = next_obs[8] - next_obs[13]   # leg1_contact - leg2_contact
    gait_bonus = 0.03 * contact_diff ** 2

    # ---------- 动作效率信号 ----------
    # 每步轻微惩罚力矩平方和，激励节能步态。
    action_penalty = -0.02 * (action[0] ** 2 + action[1] ** 2 +
                              action[2] ** 2 + action[3] ** 2)

    # ---------- 极轻微垂直速度惩罚 ----------
    vertical_velocity = next_obs[3]
    vertical_penalty = -0.03 * vertical_velocity ** 2

    # ---------- 总奖励 ----------
    total_reward = velocity_gated + gait_bonus + action_penalty + vertical_penalty
    components = {
        "velocity_gated": velocity_gated,
        "gait_bonus": gait_bonus,
        "action_penalty": action_penalty,
        "vertical_penalty": vertical_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 倾倒前兆由姿态门控覆盖，主速度信号直接瞄准任务目标，动作效率与步态交替填补了能耗与步态品质缺口。
- **behavior**: agent 已能稳定高速行走（score 320.8，len 891），步态交替奖励可进一步优化节奏、减少无用双支撑期。
- **signal**: 欠缺步态交替反馈，新增基于触地差异的连续正奖励。
- **level**: Level 3（全新骨架，保留已验证的核心结构并扩展新信号）。
- **hypothesis**: 微小交替奖励会引导清晰的左右交替模式，提高速度一致性与能源效率，有望继续提升得分。
- **risk**: 若系数过大可能诱使 agent 刻意制造单腿支撑而忽略稳定，0.03 的轻量系数将此风险降到最低。