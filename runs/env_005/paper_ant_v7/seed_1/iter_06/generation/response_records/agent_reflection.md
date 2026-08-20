# Response Record

# 设计理由
当前奖励函数的主要缺陷是 **health_gate 经常塌缩到 0**，导致 forward_progress 组件在绝大多数步上没有非零梯度（active_rate 仅 14.3%）。agent 在训练过程中无法获得持续的前进引导信号，只能依靠微弱的侧向惩罚进行探索，最终收敛到某种“安全但静止”的策略（存活满 1000 步，但几乎不前进，且外部评估得分极低）。

本轮进行 **Level 2 结构变换**：将 health_gate 中的 height_factor 和 upright_factor 的裁剪下界从 0 提升到 0.1，确保即使在接近终止边界的状态下，progress 信号仍然存在（因子 ≥0.1），从而保留梯度让 agent 有能力自我修正。同时保留因子在安全区的完整奖励（1.0），维持对良好行为的强激励。

其他系数保持原样，不引入新惩罚，避免干扰。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals
    body_z = next_obs[0]
    qx = next_obs[2]
    qy = next_obs[3]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]

    # 1. Health gate: height factor (safe range [0.3, 0.9] with soft falloffs)
    # Lower bound: clip to at least 0.1 so gradient never fully disappears
    if body_z < 0.3:
        height_factor = (body_z - 0.2) / 0.1   # reaches 0 at z=0.2
    elif body_z > 0.9:
        height_factor = (1.0 - body_z) / 0.1   # reaches 0 at z=1.0
    else:
        height_factor = 1.0
    height_factor = max(0.1, min(1.0, height_factor))

    # 2. Health gate: upright factor (from body_up_z = 1 - 2*(qx^2 + qy^2))
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    # Map [ -0.2 , 0.5 ] to [0, 1], with floor 0.1
    upright_factor = (body_up_z + 0.2) / 0.7
    upright_factor = max(0.1, min(1.0, upright_factor))

    health_gate = height_factor * upright_factor

    # 3. Main learning signal: squared forward velocity, gated by health
    forward_speed = max(0.0, body_x_vel)
    progress_reward = 2.0 * (forward_speed ** 2) * health_gate

    # 4. Light regularization against lateral drift
    lateral_penalty = -0.5 * (body_y_vel ** 2)

    total_reward = progress_reward + lateral_penalty
    components = {
        'forward_progress': progress_reward,
        'lateral_penalty': lateral_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: health_gate 易塌缩为 0，导致主进度信号在 85.7% 的步中消失，梯度大面积平坦，agent 无法学习前进。
- **behavior**: agent 存活满 1000 步但不前进（len=1000, terminated=0），在安全边界附近缓慢徘徊，几乎得不到任何前进奖励。
- **signal**: 缺少“边界附近仍有梯度”的 safety margin；gate 下界为 0 造成探索盲区。
- **level**: Level 2
- **hypothesis**: 将两个 gate 因子的裁剪下界提高到 0.1 后，agent 在所有状态下都能获得 attenuated progress 信号，从而能由梯度引导回安全区并学会前进，forward_progress 的 active_rate 预计将大幅上升，最终提升真实环境得分。
- **risk**: 下界 0.1 可能让 agent 在极度危险的状态（如 z 接近 0.2）仍获得少量 progress 奖励，从而略微延迟它学习完全避开危险区的速度；但整体上梯度引导的收益远大于此风险。若出现保持“骑在边界上”的 exploit，可在后续轮次中微调系数。
