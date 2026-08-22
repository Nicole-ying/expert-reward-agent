# 设计理由
当前 `hinge_balance_penalty` 是僵尸组件（active_rate=1.7%），对训练几乎无贡献。行为诊断表明 agent 在存活 216 步后摔倒，但倾角信号（hull_angle）很少触发惩罚，说明摔倒的直接诱因可能不是持续大倾角，而是冲击性弹跳或双脚离地导致的失控。观测中 `vertical_speed`（obs[3]）和地面接触（obs[12]/[13]）未被使用，其中垂直速度能捕捉弹跳冲击——这是 iter2 尝试过但惩罚过强失败的方向。本次用**极轻量带 safe‑zone 的 hinge**替换僵尸组件，仅对超过 0.5 m/s 的垂直速度施加微小惩罚，避免抑制正常行走起伏，同时给 agent 一个“避免跳跃”的弱信号，以期减少摔倒、延长存活。

数学形式：`excess = max(0, abs(vertical_speed) - 0.5)`，`penalty = -0.15 * excess`。  
系数校准：主信号 `progress` per‑step ≈ 0.33，新惩罚 per‑step 预估 ≤ 0.05，远低于 0.3× 上限；总惩罚负担（action + vertical）依然很低，不会重蹈 iter2 的存活崩塌。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract useful signals ----------
    horizontal_speed = obs[2]
    vertical_speed = obs[3]
    hull_angle = obs[0]
    hull_angular_velocity = obs[1]

    # ---------- health gate: close to 1 when upright, decays when tilting ----------
    denom = 1.0 + 10.0 * hull_angle * hull_angle + 0.1 * hull_angular_velocity * hull_angular_velocity
    health_gate = 1.0 / denom

    # ---------- forward progress ----------
    fwd_speed = max(0.0, horizontal_speed)
    progress_component = 1.0 * fwd_speed * health_gate

    # ---------- action regularisation ----------
    action_sum_sq = action[0]*action[0] + action[1]*action[1] + action[2]*action[2] + action[3]*action[3]
    action_penalty = -0.01 * action_sum_sq

    # ---------- vertical speed hinge penalty (replaces zombie hinge_balance_penalty) ----------
    # Suppress dangerous bouncing / sudden drops while allowing normal walking vertical motion.
    vertical_safe_threshold = 0.5  # m/s, ~walking vertical limit
    excess_vertical = max(0.0, abs(vertical_speed) - vertical_safe_threshold)
    vertical_hinge_penalty = -0.15 * excess_vertical

    # ---------- total reward ----------
    total_reward = progress_component + action_penalty + vertical_hinge_penalty

    components = {
        "progress": progress_component,
        "action_penalty": action_penalty,
        "vertical_hinge_penalty": vertical_hinge_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 未使用的 `vertical_speed` 可解释部分摔倒（弹跳致死），僵尸 `hinge_balance_penalty` 几乎无响应，需要换成低强度弹跳抑制。
- **behavior**: agent 在 216 步附近摔倒，动作短暂、动作惩罚满开，弹跳可能是主因。
- **signal**: 缺少对垂直冲击的反馈，姿态惩罚因 safe zone 太宽而失效。
- **level**: Level 2
- **hypothesis**: 微小弹跳惩罚将降低冲击性摔倒，延长存活，同时不抑制有益步态，从而提升积分。
- **risk**: 若阈值/系数仍不匹配，active_rate 可能再次偏低；若惩罚过强可能复现 iter2 的存活暴跌，但本次系数极保守，风险可控。