# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next observation
    forward_vel = next_obs[5]    # forward velocity of the torso
    torso_angle = next_obs[1]    # torso tilt angle
    torso_angvel = next_obs[7]   # torso angular velocity

    # ---------- Main progress signal ----------
    # Encourage sustained forward motion. This is the primary task driver.
    w_forward = 2.0
    forward_reward = w_forward * forward_vel

    # ---------- Stability constraint ----------
    # Penalise large torso tilt and high angular velocity to keep the body upright
    # and avoid rapid tumbling.  Squared terms create a soft, continuous penalty.
    w_angle = 0.5
    w_angvel = 0.3
    upright_penalty = -w_angle * (torso_angle ** 2) - w_angvel * (torso_angvel ** 2)

    # ---------- Total reward ----------
    total_reward = forward_reward + upright_penalty

    components = {
        "forward_reward": forward_reward,
        "upright_penalty": upright_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 组件与角色

- **forward_reward（主学习信号）**：直接奖励躯干前向速度。这是一个稠密、每步可微的信号，是推进 agent 向前跳跃的核心动力。它告诉策略“向前移动就得分”。
- **upright_penalty（稳定/安全约束）**：通过惩罚躯干倾角的平方和角速度的平方，抑制过度倾斜与快速旋转，帮助 agent 保持直立姿势，降低因摔倒终止的概率。

两个组件共同定义了任务驱动与姿态约束的平衡，符合 v1 阶段“先学会向前跳跃并站稳”的目标。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡片明确声明 `explicit_success_flag_available = false` 和 `explicit_failure_flag_available = false`，且 `info` 为空，无法获取任何终止原因或成功标识。
- 缺乏显式标志时，直接使用稀疏终端奖惩极易误导训练，因此 v1 完全放弃这两个组件。

## 为什么没有使用其他常见骨架

- **alive_bonus**：需要 `done` 标志，本环境不提供，无法使用。
- **energy_penalty**：动作幅度惩罚。v1 阶段的首要目标是让 agent 学会稳定前进跳跃，过早施加能耗约束可能导致 agent 不敢用力，出现“站着不动”的保守策略。该组件计划在后续迭代（v2+）引入。
- **action_smoothness_penalty**：同样属于效率/平滑类需求，v1 先聚焦核心运动，后续若出现动作抖动再考虑加入。
- **height 惩罚**：环境未给出健康高度范围的目标值或阈值，硬编码一个参考高度风险较大（可能错误压制必要的跳跃高度），故暂不纳入，由 upright_penalty 间接防止摔倒。

## 计划后续迭代加入的组件

- 若出现 **fast_then_fail**（跑得快但经常摔倒），可在 v2 中加大 upright_penalty 权重，或加入对躯干高度的软约束（如惩罚高度低于某经验阈值）。
- 若策略学会跳跃后稳定但能耗过高，可加入 **energy_penalty**（动作幅度惩罚）。
- 若有周期性抖动或动作不平滑，可加入 **action_smoothness_penalty**。

## 训练后应观察的 failure mode

- **hover_or_stand_still**：forward_vel 接近零，agent 可能依赖微小的 forward_vel 波动获取少量 reward（概率低，因为正向奖励仅当速度为正时出现，速度很小则奖励很小）。若出现，检查向前速度奖励的尺度是否不足，或 upright_penalty 是否过强抑制了主动跌倒前的试探。
- **fast_then_fail**：forward_vel 很大但频繁因倾角/高度超出范围终止，说明稳定性惩罚权重不足，需上调。
- **over_conservative_policy**：动作幅度极小，倾斜严格受控但前进缓慢，表示 upright_penalty 过重。此时可降低其权重或在后续迭代中引入对前进速度的更强激励。
