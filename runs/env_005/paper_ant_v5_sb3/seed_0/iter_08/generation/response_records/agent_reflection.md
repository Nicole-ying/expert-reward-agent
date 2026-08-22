# Response Record

# 设计理由
terminated 比例高达 90%，证明大多数 episode 因身体高度越界（body_z < 0.2）失败，而当前 upright_pen 只惩罚倾斜，却未能有效阻止坠落。我选择将干预目标从直立惩罚改为直接的身体高度铰链惩罚（proxy 对齐任务完成），阈值设在终止边界 0.2 的 60-80% 处（0.3），系数 1.0 使得 body_z 逼近 0.2 时 per‑step 惩罚约 0.1，与典型 forward 奖励（per‑step ≈ 0.45）构成合理制衡，避免过度负馈。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    body_x_vel = next_obs[13]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_z = next_obs[0]

    # 直立因子（body_up_z ∈ [-1, 1]，1 为全直立）
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_factor = max(0.0, body_up_z)

    # 前向奖励（仅当身体朝上时）
    forward = 1.0 * body_x_vel * upright_factor

    # 身体高度铰链惩罚：在高度接近 0.2 前发出警告
    height_threshold = 0.3
    w_height_pen = 1.0
    height_pen = -w_height_pen * max(0.0, height_threshold - body_z)

    total_reward = forward + height_pen

    components = {
        "forward": forward,
        "height_pen": height_pen
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated 18/20，平均长度 167 步，upright_pen 占 magnitude_share 71.3% 且 active_rate 57.7%，直接终止原因为 body_z 越界。
- **behavior**: agent 勉强前进（forward 约 0.45 per‑step）但频繁跌倒，铰链惩罚过大导致总奖励为负。
- **signal**: 缺少针对 body_z 的直接保护梯度，upright_pen 偏离真实终止原因。
- **level**: Level 2 — proxy 与外部任务错位，将惩罚目标从直立改为身体高度。
- **hypothesis**: 直接惩罚 body_z 接近 0.2 的区域能更早阻止坠落并给出精确恢复信号，保留 forward×upright_factor 维持直立引导。
- **risk**: 移除直立惩罚可能让 agent 暂时倾向倾斜，但高度惩罚会通过迫使抬升高度间接要求身体回正，初期可能出现短暂性能波动。
