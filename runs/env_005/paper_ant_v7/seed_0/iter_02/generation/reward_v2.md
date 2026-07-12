# 设计理由
当前 agent 在 80 步左右因高度越界（低于 0.2 或高于 1.0）而终止，生存时间短，累积的正向速度奖励完全被少量负惩罚吞没。原有 `height_penalty` 只是一个微弱的独立惩罚（active_rate 仅 4.8%，每步约 -0.02），在高度接近边界时无法发出足够强的纠正信号。  
改用 **软高度门控**（soft health gate）：将高度安全因子直接乘在前进奖励上，这样一旦身体高度接近危险区，前进奖励立即衰减，agent 必须维持安全高度才能获得速度回报。这属于 Level 2 结构变换——从独立约束惩罚转变为乘积 gate。  
系数校准：`height_factor` 在 [0.3, 0.9] 保持 1.0，在 [0.2, 0.3] 和 [0.9, 1.0] 线性衰减到 0，终止边界处 gate=0。不引入额外惩罚负担，避免惩罚累积压制探索。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    torque_penalty = sum([a**2 for a in action])

    # ---------- 高度安全门控 ----------
    # [0.2, 0.3] 线性 0→1, [0.9, 1.0] 线性 1→0
    low_factor = max(0.0, (body_z - 0.2) / 0.1)
    high_factor = max(0.0, (1.0 - body_z) / 0.1)
    height_factor = min(low_factor, high_factor)  # 安全区=1, 越危险→0

    # ---------- 主学习信号：被高度门控的前进速度 ----------
    forward_reward = 1.0 * body_x_velocity * height_factor

    # ---------- 直立姿态约束（二次惩罚） ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_error = 1.0 - body_up_z
    upright_penalty = -2.0 * (upright_error**2)

    # ---------- 力矩效率约束 ----------
    action_cost = -0.01 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = forward_reward + upright_penalty + action_cost

    components = {
        "forward_reward": forward_reward,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 高度约束太弱（penalty 仅有 4.8% active，幅度可忽略），缺少角速度稳定性惩罚但当前优先解决生存问题。
- **behavior**: agent 在约 80 步时因高度越界失败，生存时间短，累积 speed reward 不足以抵消负分。
- **signal**: 高度约束信号失效，未能提前阻止危险动作。
- **level**: Level 2（高度从独立惩罚改为乘积门控）
- **hypothesis**: 乘性高度门使速度奖励与生存强绑定，agent 会主动保持安全高度以获取前进回报，生存时间延长，累积 speed 得分应大幅提升。
- **risk**: agent 可能通过原地踏步或慢速移动来维持高度安全，导致速度偏低；若发生此现象，后续可增加凸化速度奖励或横向摆动惩罚。