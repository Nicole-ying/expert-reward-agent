# Response Record

# 设计理由
移除了上一轮引入的 `height_gate`（它导致 score 从 263.65 暴跌到 -53.61），回到只保留前进奖励与姿态惩罚的结构。对姿态惩罚项做 **Level 2 数学结构变换**：将全时二次惩罚改为 **hinge 惩罚**，只在 `body_up_z < 0.5` 时生效（阈值设在直立值 1.0 的 50% 处，给 agent 足够的警告区间），惩罚使用线性偏差，系数 `1.5` 使危险区 per‑step 惩罚量级约为主前进信号的 0.1‑0.3 倍，避免主导但足以引导恢复直立。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_x_vel = next_obs[13]
    quat_x = next_obs[2]
    quat_y = next_obs[3]

    # 直立因子（body_up_z ∈ [-1, 1]，1 为全直立）
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, body_up_z)

    # 前向奖励（仅当身体朝上时）
    forward = 1.0 * body_x_vel * upright_factor

    # 直立偏差 hinge 惩罚（只在倾斜过大时激活）
    threshold = 0.5
    w_hinge = 1.5
    upright_pen = -w_hinge * max(0.0, threshold - body_up_z)

    total_reward = forward + upright_pen

    components = {
        "forward": forward,
        "upright_pen": upright_pen
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: 上一轮引入 height_gate 后外部 score 从 263.65 暴跌至 -53.61，且 terminated=19/20；upright_pen 虽然 100% 活跃但未能阻止摔倒。
- **behavior**: agent 在多数 episode 中发生致命倾斜（导致跌倒），现有均匀姿态惩罚梯度不足以防止危险姿态。
- **signal**: 全时二次姿态惩罚未能区分安全与危险区，危险区梯度太弱。
- **level**: Level 2 —— 保留前进+姿态职责，但将惩罚形态改为 hinge。
- **hypothesis**: hinge 只在倾斜超过阈值时施加强信号，能让 agent 更早学习恢复直立，同时去除 gate 消除上轮引入的干扰，有望回到并超越之前的 high-score 策略。
- **risk**: 阈值 0.5 可能过高，过早惩罚会抑制对快速转向的探索，导致运动过于保守，降低前进速度。
