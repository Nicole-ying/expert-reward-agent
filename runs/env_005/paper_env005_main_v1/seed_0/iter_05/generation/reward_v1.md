# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Materially different reward design (v1) — no height-dependent terms.
    Focuses purely on forward motion, upright orientation, and suppressing
    lateral drift and angular rates, without any explicit height penalty.
    """
    # next_obs indices:
    BODY_X_VEL = 13
    BODY_Y_VEL = 14
    ROLL_VEL = 16
    PITCH_VEL = 17

    # Quaternion components: assumed order (w, x, y, z)
    # Body up direction z-component in world frame
    qw = next_obs[1]
    qx = next_obs[2]
    qy = next_obs[3]
    qz = next_obs[4]  # not used for body_up_z but included for completeness
    body_up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

    forward_vel = next_obs[BODY_X_VEL]
    lateral_vel = next_obs[BODY_Y_VEL]
    roll_vel = next_obs[ROLL_VEL]
    pitch_vel = next_obs[PITCH_VEL]

    # Weights — chosen to keep forward reward dominant
    w_fwd   = 1.0
    w_orient = 0.3
    w_lat   = 0.2
    w_ang   = 0.05

    # Components
    forward_reward = w_fwd * forward_vel
    orientation_reward = w_orient * body_up_z
    lateral_penalty = -w_lat * abs(lateral_vel)
    angular_penalty = -w_ang * (abs(roll_vel) + abs(pitch_vel))

    total_reward = forward_reward + orientation_reward + lateral_penalty + angular_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "orientation_reward": orientation_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **forward_velocity_reward**（主学习信号）  
   直接奖励世界坐标系下的前向速度 `body_x_velocity`。告诉 agent 持续向前移动是最根本的目标。

2. **orientation_reward**（稳定/姿态约束）  
   利用四元数计算身体上方向在世界坐标系中的投影 `body_up_z`，奖励直立姿态（值接近 1）。该连续信号提供每步的梯度，帮助 agent 避免翻滚。

3. **lateral_velocity_penalty**（稳定/漂移约束）  
   惩罚侧向速度的绝对值。抑制横向滑动，使运动方向更集中于前进方向，有利于直线稳定行走。

4. **angular_velocity_penalty**（稳定/角速度约束）  
   惩罚身体滚动和俯仰角速度的大小。在失衡过程中角速度往往急剧增大，施加惩罚可以提前阻止倾倒趋势。

## 设计假设及与前序尝试的差异

- 前序尝试都以某种形式包含**高度相关项**（如高度惩罚、高度 hinge loss、高度门控等），但反复调参和门控变换仍未解决崩塌问题，最优得分仅 -49.53。  
- 本次设计**完全移除高度维度**——不奖励、不惩罚任何身体高度信号。  
  假设高度惩罚或高度门控可能干扰了姿态稳定策略的学习，移除后 agent 将仅通过姿态和角速度约束来学习防倒，而高度仅受环境终止边界保护。  
- 新增 **lateral_velocity_penalty**，这是前面结构从未单独强调的组件，能进一步抑制漂移、促进稳健的前向步态。  
- 整体结构为 **1 个主学习信号 + 3 个轻量稳定约束**，符合 v1 阶段的 role‑based budget，并且数学形态均为连续函数、提供每步梯度，无二值门控，易于早期学习。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境中不存在明确的成功标志 (`explicit_success_flag_available = false`)，info 字典也为空，无法获取 termination reason。  
- 虽然可以通 `next_obs[0]` 推断部分失败终止，但专家指令明确禁止在无显式标志时使用 `terminal_failure_penalty`，因此不在 v1 中加入。

## 留到后续迭代的组件

- **alive_bonus / 高度相关的奖励**：若后续实验表明 agent 因缺乏高度引导而频繁擦地或低姿态行进，可重新引入柔性的存活激励或高度中心奖励，但前提是不引起新的崩塌模式。  
- **能量/动作平滑 penalty**：待 agent 学会稳定前进后再加入，避免过早压制探索。  
- **步态 / 足部接触相位奖励**：有可能引入更结构化的行走周期引导，但也可能增加复杂度，需要 v2、v3 阶段谨慎评估。

## 训练后应重点观察的 failure mode

1. **低姿态行走（low walk）** – 机器人可能用非常低的底盘前进，短期不倒但运动缓慢、效率低下。若出现，需考虑引入温和的高度引导。  
2. **侧向振荡** – 若 `lateral_velocity_penalty` 权重过小，可能出现左右摇晃式前进。可调大 `w_lat` 并观察。  
3. **过于保守** – 若 `angular_velocity_penalty` 太强，agent 可能不敢迈步只能缓慢挪动。可通过调低 `w_ang` 或提升 `w_fwd` 激励行动。  
4. **高频抖动** – 连续运动但关节高速抖动。此版本未加动作平滑，未来可引入 `action_smoothness_penalty`。