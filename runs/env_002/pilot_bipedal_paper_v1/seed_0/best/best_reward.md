## 1. Evidence
Score=150.58，episode平均长度756步，全部terminated（无truncation），其中7/20（35%）在150步内以score<-50提前终止，剩余13条轨迹存活至约1109步、score约275；forward_reward占signed share 89%、magnitude share 90%，stability_cost仅占10%，两者比值约0.11，属于提示中"轻约束"下界。

## 2. Behavior Diagnosis
策略呈双峰行为：65%的episode能稳定行走并获得较高得分（~275），但35%的episode快速摔倒（score<-50、长度<150步）。前向速度激励占绝对主导，稳定性约束过轻，导致策略在某些初始状态或扰动下无法维持平衡，直接摔倒。

## 3. Signal Completeness
必要职责中，前向进展（forward_velocity）和基本稳定性（倾角、角速度、垂直速度）已覆盖，但缺少能量效率信号（任务明确要求"尽量减小能量消耗"）。当前最紧迫的问题是稳定性约束偏弱导致的高早期终止率，而非缺失的能量信号。

## 4. Selected Level
**Level 1（尺度修复）**：稳定性组件职责和数学形态合理（惩罚倾角、角速度、垂直振荡），但相对前向奖励过弱，|stability|/|forward|≈0.11仅处于轻约束下界，而35%早期摔倒率直接证明稳定性约束不足。

## 5. Selected Intervention
唯一目标组件：`stability_cost`。将其三个子系数翻倍——`w_angle`: 0.5→1.0，`w_angvel`: 0.1→0.2，`w_vvel`: 0.2→0.4。`forward_reward`系数保持不变。预期|stability|/|forward|升至约0.22，仍在0.5警戒线以下。

## 6. Falsifiable Hypothesis
翻倍稳定性系数应使agent在危险姿态（大倾角、高角速度、剧烈垂直振荡）时受到更强负反馈，降低摔倒概率，从而减少早期终止比例（从35%下降），提升平均得分并缩小score_range。

## 7. Expected Next Round
- 早期终止率（<150步且score<-50）应从7/20下降
- score_range应缩小（最低分提升）
- stability_cost的magnitude_share应从10%上升至约18-22%
- 平均score应提升（减少灾难性负分拖累）
- episode_length应略微上升或保持

## 8. Main Risk
稳定性约束过强可能抑制步态探索，导致策略偏向保守（小步幅、低速度），forward_reward下降；若出现此情况需降回系数或转Level 2改用非线性形式。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励
    forward_velocity = next_obs[2]
    w_forward = 1.0
    forward_reward = w_forward * forward_velocity

    # 稳定性约束：惩罚身体倾角、角速度和剧烈垂直振荡（系数翻倍）
    hull_angle = next_obs[0]
    hull_angular_vel = next_obs[1]
    vertical_vel = next_obs[3]

    w_angle = 1.0
    w_angvel = 0.2
    w_vvel = 0.4
    stability_cost = (
        -w_angle * abs(hull_angle)
        - w_angvel * abs(hull_angular_vel)
        - w_vvel * abs(vertical_vel)
    )

    total_reward = forward_reward + stability_cost

    components = {
        "forward_reward": forward_reward,
        "stability_cost": stability_cost,
    }

    return float(total_reward), components
```