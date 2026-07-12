# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 中提取关键信号
    body_z = next_obs[0]         # 身体高度
    quat_x = next_obs[2]         # 四元数虚部 x
    quat_y = next_obs[3]         # 四元数虚部 y
    quat_z = next_obs[4]         # 四元数虚部 z
    vx = next_obs[13]            # 世界系前向速度
    vy = next_obs[14]            # 世界系侧向速度

    # 1. 前向速度奖励（主学习信号）：只奖励非负的前向速度
    forward_reward = 1.0 * max(0.0, vx)

    # 2. 侧向速度惩罚：抑制横向漂移
    lateral_penalty = -0.5 * (vy ** 2)

    # 3. 身体高度安全约束：只在高度越出 [0.2, 1.0] 时进行线性惩罚
    height_low = 0.2
    height_high = 1.0
    height_exceed = max(0.0, height_low - body_z) + max(0.0, body_z - height_high)
    height_penalty = -10.0 * height_exceed

    # 4. 姿态直立约束：惩罚四元数虚部分量平方和，推动 quat_w 趋近 1
    pose_error = quat_x**2 + quat_y**2 + quat_z**2
    pose_penalty = -5.0 * pose_error

    # 总奖励
    total_reward = forward_reward + lateral_penalty + height_penalty + pose_penalty

    # 组件字典
    components = {
        "forward_reward": forward_reward,
        "lateral_penalty": lateral_penalty,
        "height_penalty": height_penalty,
        "posture_penalty": pose_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype

- **task_family:** locomotion_continuous_control  
- **dynamics_subtype:** quadruped_walking (3D 四足机器人在地面上持续向前推进)

## selected reward roles

| Role | Component | 动机 |
|---|---|---|
| **主学习信号** (mandatory) | forward_reward = `max(0, vx)`  | 驱动机器人产生持续正向速度，是任务核心目标 |
| **稳定性约束** (conditional) | lateral_penalty = `-vy²` | 抑制侧向漂移，鼓励直线前进，减少无用功耗 |
| **健康约束** (conditional) | height_penalty = hinge 惩罚超出 `[0.2, 1.0]` 的部分 | 防止摔倒（过低）或异常跃起（过高），仅在边界外生效 |
| **姿态约束** (conditional) | pose_penalty = 四元数虚部平方和 | 保持躯干直立，避免翻滚/俯仰/偏航过大导致失稳 |

## role_to_signal_mapping

- forward_reward → `next_obs[13]` (`body_x_velocity`)  
- lateral_penalty → `next_obs[14]` (`body_y_velocity`)  
- height_penalty → `next_obs[0]` (`body_z`)  
- pose_penalty → `next_obs[2:5]` (`quat_x, quat_y, quat_z`)

## 每个 role 选择的 formula operator

- **forward_reward:** `dense_state_signal` 线性正奖励，配合 `max(0,·)` 避免奖励倒退。  
- **lateral_penalty:** `quadratic_penalty` (二次惩罚) 对侧向速度敏感，抑制大幅偏移。  
- **height_penalty:** `dense_state_signal` 的 hinge 形式，只在安全区间外线性惩罚，安全区间内不干扰正常运动。  
- **pose_penalty:** `quadratic_penalty` 对姿态误差的连续二次惩罚，驱动机器人恢复直立。

## excluded roles 及原因

- **efficiency / action cost**：留给后续迭代，v1 优先学会前进与稳定。  
- **terminal_success_reward**：`explicit_success_flag_available=false`，环境没有成功标志，无法使用。  
- **terminal_failure_penalty**：`explicit_failure_flag_available=false`，info 为空，无法获取终止原因。  
- **soft_health_gate**：当前约束采用独立的 hinge/quadratic 惩罚，若后续出现“前进但常摔倒”可在未来版本改为门控乘到主信号上。  
- **joint_condition_proxy / curriculum**：v1 无需多条件联合完成判定或动态课程。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境不提供 `info` 中的 success/failure flag，且任务是无终点的连续移动，不存在天然的成功/失败终点。  
- 终止仅由高度越界或数值 NaN/inf 触发，这些信息未传入 reward 函数，因此无法构建基于终态的稀疏奖励。

## 哪些职责留到后续迭代

- 动作平滑度/能耗优化（`action_smoothness_penalty` 或 `-energy_cost`）  
- 更高阶的姿态、角速度约束（滚转/俯仰角速度直接惩罚）  
- 如果出现“先冲后死”的高终止率现象，引入 `soft_health_gate` 将高度/姿态信号作为主奖励的门控因子  
- 可能的能量效率、接触力信息（若未来环境提供）

## 训练后应该观察哪些 failure modes

- **velocity_burst_then_fall:** 机器人短暂前冲后因姿态失衡而摔倒。应检查 `pose_penalty` 和 `height_penalty` 是否及时给出了梯度。  
- **lateral_drift:** 侧向速度未被充分抑制，机器人蛇形前进。需关注 `lateral_penalty` 的有效性。  
- **stand_still_or_slow_crawl:** 前向速度很低，agent 安于小幅奖励。可能需要凸化 `forward_reward`（如 `vx²`）或增加速度下限门控。  
- **high_pose_penalty_suppresses_movement:** 姿态惩罚过强导致机器人“趴着不敢动”。若出现此情况，可降低 `pose_penalty` 权重或改用 hinge 只在倾斜超过安全角时惩罚。
