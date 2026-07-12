# 设计理由
这是首轮反思，当前奖励函数存在两个关键薄弱点：(1) 平衡惩罚 `angle_penalty` 使用二次型但系数过小，在 hull_angle 接近危险阈值（~0.5rad）时梯度不足，无法有效阻止摔倒；(2) `gate_factor` 使用线性衰减，在 0.3-0.5rad 区间 gate 已大幅衰减，压制前进动力过早。

**推测问题根因**：agent 在低速安全区（angle小，速度慢）和高速高风险区（速度快但angle摆动大）之间摇摆。当前 `balance_penalty` 的 magnitude_share 仅 0.1%，完全无法对抗前进激励，导致 agent 偏向冒险前进→摔倒。

**修改策略（Level 2 结构变换）**：
- **将 `angle_penalty` 从二次型改为 hinge**：设定阈值在终止边界 0.5rad 的 70%（≈0.35rad），让惩罚在 0.35-0.5rad 区间有更强的线性梯度，在 0-0.35rad 区间惩罚归零（鼓励正常平衡范围内的探索）。
- **同步调整 `gate_factor` 为 0.35-0.5 区间更急剧的衰减**：从 `max(0, 1 - |angle|/0.6)` 改为 `max(0, 1 - (|angle|-0.25)/(0.5-0.25))`，让 gate 在 0.25rad 后才开始衰减，在 0.5rad 归零。这样在安全区（0-0.25rad）完全不惩罚前进，在 0.25-0.5rad 渐进收紧。
- **系数校准**：主信号 `forward_reward` 的 per-step 约 0.47（381.44/820.35），hinge penalty 在 angle=0.5 时约 -0.8（系数 4.0 × (0.5-0.3)=0.8），per-step penalty burden 约为主信号的 0.1x，远低于 0.3x 上限。

修改一个组件（`angle_penalty` 变换）+ 微调一个已有组件（`gate_factor` 阈值），按 Level 2 规则视作一个结构变换。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Signal extraction
    # ============================================================
    # forward_progress: horizontal_velocity (obs[2])
    horizontal_vel = obs[2]
    
    # balance_maintenance: hull_angle (obs[0]), hull_angular_velocity (obs[1])
    hull_angle = obs[0]
    hull_angular_vel = obs[1]
    
    # ============================================================
    # Component 1: forward_progress (main learning signal)
    # Use dense_state_signal: positive linear reward for forward velocity
    # Scale: typical horizontal velocity range ~0-5 m/s, target ~2-3 m/s
    # Weight chosen so that at 2 m/s, this component ≈ 2.0
    # ============================================================
    forward_reward = 1.0 * horizontal_vel
    
    # ============================================================
    # Component 2: balance_maintenance (stability constraint)
    # CHANGED: hinge penalty for hull_angle instead of quadratic
    # hinge_threshold = 0.3 rad (~17 deg), which is 60% of termination boundary 0.5 rad
    # Penalty kicks in only when |hull_angle| > 0.3, linearly to -0.8 at 0.5 rad
    # This provides stronger gradient near danger zone while not penalizing normal sway
    # Angular velocity penalty kept mild to discourage fast wobbling
    # ============================================================
    angle_threshold = 0.3  # rad, ~17 deg — 60% of 0.5 rad termination boundary
    angle_deviation = max(0.0, abs(hull_angle) - angle_threshold)
    angle_penalty = -4.0 * angle_deviation  # at |hull_angle|=0.5: penalty ≈ -0.8
    
    angular_vel_penalty = -0.1 * (hull_angular_vel ** 2)
    balance_penalty = angle_penalty + angular_vel_penalty
    
    # ============================================================
    # Component 3: soft_health_gate on forward_reward
    # CHANGED: hinge-based gate with relaxed start threshold (0.25 rad)
    # gate = max(0, 1 - (|hull_angle| - 0.25) / (0.5 - 0.25))
    # At hull_angle <= 0.25: gate = 1.0 (full forward reward)
    # At hull_angle = 0.35: gate ≈ 0.6
    # At hull_angle = 0.5: gate = 0.0
    # ============================================================
    gate_lower = 0.25  # no reduction below this angle
    gate_upper = 0.5   # full cutoff at this angle (termination boundary)
    gate_raw = max(0.0, 1.0 - (abs(hull_angle) - gate_lower) / (gate_upper - gate_lower))
    gate_factor = gate_raw  # can't collapse at safe region: at |angle|=0.3, gate≈0.8
    
    gated_forward = forward_reward * gate_factor
    
    # ============================================================
    # Total reward
    # ============================================================
    total_reward = gated_forward + balance_penalty
    
    # ============================================================
    # Components dict (for debugging and monitoring)
    # ============================================================
    components = {
        'forward_reward': forward_reward,
        'gated_forward': gated_forward,
        'gate_factor': gate_factor,
        'angle_penalty': angle_penalty,
        'angular_vel_penalty': angular_vel_penalty,
        'balance_penalty': balance_penalty,
        'angle_deviation': angle_deviation
    }
    
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前 reward 漏了强平衡约束和动作效率信号；`balance_penalty` 幅度过弱（share 0.1%）完全无法对抗前进激励，导致 agent 频繁冒险摔倒。
- **behavior**: agent 在低速安全区和高速高风险区之间摇摆，部分 episode 因摔倒获得极低分（-43），但平均得分尚可（166），表明前进步态尚未稳定。
- **signal**: `angle_penalty` 的二次型在 0.3-0.5rad 区间梯度不足；`gate_factor` 线性衰减过早压制前进动力。
- **level**: Level 2
- **hypothesis**: 将 `angle_penalty` 改为 hinge 并在 `gate_factor` 中引入更宽的免衰减区，可以让 agent 在安全区自由探索前进，在接近危险区时获得足够强的约束信号，从而减少摔倒、延长 episode、提高最终得分。
- **risk**: hinge 在 threshold 处梯度不连续可能导致 training 初期在 0.3rad 附近有微弱的震荡行为；但该风险远小于当前频繁摔倒的主问题。