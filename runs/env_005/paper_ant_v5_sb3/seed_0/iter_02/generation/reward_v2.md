# 设计理由
上一轮 upright 惩罚系数过大（10），在 uiright_error 平均约 0.2 时，每步惩罚约 -0.4，远高于 forward 的每步约 +0.13，导致总奖励被压制，agent 不敢探索前进动作。本次将 w_upright 从 10 降至 1.5，使惩罚量级回到 forward 的 0.3 倍附近，恢复 forward 信号的主导地位，同时保留足够的直立倾向。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    body_z = next_obs[0]
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]          # not used but available
    body_x_vel = next_obs[13]

    # 1. Forward progress: reward positive forward velocity
    w_forward = 1.0
    forward = w_forward * body_x_vel    # negative when moving backward (natural penalty)

    # 2. Upright orientation: penalty for deviation from perfectly upright
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    upright_error = 1.0 - body_up_z
    w_upright = 1.5                   # reduced from 10.0
    upright_pen = -w_upright * upright_error**2

    # 3. Healthy body height: hinge penalty only when approaching termination bounds
    w_height = 5.0
    height_pen_low = max(0.0, 0.25 - body_z)
    height_pen_high = max(0.0, body_z - 0.95)
    height_pen = -w_height * (height_pen_low + height_pen_high)

    total_reward = forward + upright_pen + height_pen

    components = {
        "forward": forward,
        "upright": upright_pen,
        "height": height_pen
    }

    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated 14/20，score=-371.5，forward 每步约 0.13，upright 每步约 -0.4，信号严重失衡。
- **behavior**: 多数 episode 以失败终止，agent 倾向静止或轻微移动后跌倒，无法稳定前进。
- **signal**: upright 惩罚过强，完全压倒 forward，导致探索被抑制，前进信号无法引导行为。
- **level**: Level 1（尺度修复），触发条件：|penalty/progress| = 3 > 0.5 且惩罚主导总奖励。
- **hypothesis**: 降低 upright 系数后，forward 信号主导，agent 更愿意尝试前进动作，从而有机会学习动态平衡。
- **risk**: 直立约束变弱可能导致短暂倾斜幅度增大，但依然有 hinge 高度惩罚和 upright 惩罚兜底，不会立即崩溃。