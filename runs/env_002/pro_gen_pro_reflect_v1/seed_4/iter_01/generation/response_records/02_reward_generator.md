# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- 核心观测提取 --------------------
    hull_angle     = float(obs[0])   # 身体倾角 (rad)
    hull_ang_vel   = float(obs[1])   # 身体角速度 (rad/s)
    horiz_vel      = float(obs[2])   # 前向速度 (m/s)
    vert_vel       = float(obs[3])   # 竖直速度 (m/s)

    # -------------------- 主进展奖励 (核心驱动力) --------------------
    # 只奖励正的前进速度，负值不惩罚也不奖励
    forward_reward = 1.0 * max(0.0, horiz_vel)

    # -------------------- 姿态与稳定性约束 --------------------
    # 倾角 hinge 惩罚：只在 |angle| > 0.5 rad 时生效，避免安全范围内全时惩罚
    angle_threshold = 0.5
    angle_penalty = -0.5 * max(0.0, abs(hull_angle) - angle_threshold)

    # 角速度二次惩罚：始终有轻量梯度，抑制剧烈摇摆
    angvel_penalty = -0.001 * (hull_ang_vel ** 2)

    # -------------------- 竖直速度抑制（防跳跃） --------------------
    # 步行中正常竖直速度波动 ≤0.5 m/s，超过部分轻罚，避免 agent 通过跳跃刷分
    vert_threshold = 0.5
    vert_penalty = -0.1 * max(0.0, abs(vert_vel) - vert_threshold)

    # -------------------- 总奖励 --------------------
    total_reward = forward_reward + angle_penalty + angvel_penalty + vert_penalty

    components = {
        "forward_reward": forward_reward,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "vert_penalty": vert_penalty,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务与环境信息
- selected task_family: `locomotion_continuous_control`  
- dynamics_subtype: `planar_bipedal_gait`  
- morphology: 双腿，髋膝力矩驱动  
- 主要目标：持续快速向前行走，同时避免摔倒  
- 可用信号：obs 维度 0–13 描述身体与关节状态，info 为空，无 success/failure 标志  

## 2. 选择的奖励职责及信号映射
| 职责角色 | 信号 | 公式算子 | 备注 |
|---|---|---|---|
| **forward_progress** (主学习信号) | `obs[2]` 水平速度 | `dense_state_signal` (positive 线性，带 relu 截断负值) | 奖励正速度，不惩罚后退，提供每步梯度 |
| **upright_balance_survival** (稳定约束) | `obs[0]` 倾角 → hinge penalty<br>`obs[1]` 角速度 → quadratic penalty | `dense_state_signal` 的 hinge (`max(0,|angle|-0.5)`) 和 `quadratic_penalty` | 只在危险姿态区域施加惩罚，避免安全姿态内的无意义干扰；角速度轻量抑制摇摆 |
| **anti-bounce** (辅助稳定) | `obs[3]` 竖直速度 → hinge penalty | `dense_state_signal` 的 hinge (`max(0,|vert|-0.5)`) | 防止 agent 通过跳跃获取瞬时前向速度，但正常行走的竖直波动不受罚 |

## 3. 排除的职责及原因
- `energy_efficiency`、`action_smoothness`、`contact_gait_rhythm`：v1 阶段暂不引入，避免过早压制探索。 待 agent 学会稳定前进后再逐步加入，目前以“走起来且不摔倒”为首要目标。
- `terrain_awareness_from_lidar`：激光雷达维度高且缺少可解释的标量抽取，基础奖励设计中不需要。
- `terminal_sparse_success`：info 字典为空，无到达终点或摔倒标记，无法可靠使用稀疏奖励。

## 4. 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境未提供任何显式成功/失败标志（`explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，`info={}`），因此无法实现基于回合结束信号的稀疏奖励。所有学习信号必须来自逐步的观测值。

## 5. 后续迭代预留给的职责
- **energy_efficiency**：当 agent 能稳定前进 >0.5 m/s 时，逐步引入动作力矩的二次惩罚，权重由低升高。
- **action_smoothness**：若出现关节震颤，缓存上一动作并惩罚动作差分的 L2 范数。
- **contact_gait_rhythm**：若步态异常（如双脚长期腾空或拖沓不交替），可加入对接触模式的轻量奖励引导。

## 6. 训练后需观察的 failure modes
1. **agent 不敢动（全零动作）**：`forward_reward≈0`，episode 短。检查是否角度/角速度惩罚过重导致每步总奖励为负。  
2. **快速摔倒（倾角大或角速度发散）**：`angle_penalty` 与 `angvel_penalty` 绝对值很大，但未阻止摔倒。可能需要调整 hinge 的 threshold 或适当加大惩罚系数（如 angle_penalty 乘子从 0.5 上调至 1.0）。  
3. **原地踏步但无前进**：`horizontal_velocity` 均值为零，动作不为零。可能需要对水平速度使用凸化形式（例如 `max(0,horiz_vel)**2`）或加入接触节奏引导。  
4. **跳跃式前进**：若 `vert_velocity` 频繁超过阈值，且 agent 通过跳跃获取前进速度，可适当提高 `vert_penalty` 系数或降低阈值到 0.3 rad/m。
