# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next_obs
    body_z = next_obs[0]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]

    # Body uprightness (1.0 = perfectly upright, 0.0 = tilted)
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    # Guard against tiny numerical overshoot
    body_up_z = max(0.0, min(1.0, body_up_z))

    # ---------- Forward progress (bounded, only positive velocity) ----------
    vx = max(0.0, body_x_vel)
    forward_reward = vx / (1.0 + vx)          # bounded in [0, 1), strong gradient at low speed

    # ---------- Height safety: hinge penalty near termination boundaries ----------
    height_low_violation  = max(0.0, 0.3 - body_z)
    height_high_violation = max(0.0, body_z - 0.9)
    height_penalty = height_low_violation**2 + height_high_violation**2

    # ---------- Upright posture penalty ----------
    upright_penalty = (1.0 - body_up_z)**2

    # ---------- Lateral slip penalty ----------
    lateral_penalty = abs(body_y_vel)

    # Weights (balanced so that normal walking yields positive total reward)
    w_forward  = 1.0
    w_height   = 5.0
    w_upright  = 5.0
    w_lateral  = 0.5

    total_reward = (w_forward * forward_reward
                    - w_height   * height_penalty
                    - w_upright  * upright_penalty
                    - w_lateral  * lateral_penalty)

    components = {
        "forward_reward": w_forward * forward_reward,
        "height_penalty": w_height * height_penalty,
        "upright_penalty": w_upright * upright_penalty,
        "lateral_penalty": w_lateral * lateral_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与职责映射
- **task_family / dynamics_subtype**: `locomotion_continuous_control` / `multi_legged_body_locomotion`  
- **selected reward roles (mandatory → conditional)**:
  1. `forward_progress` – 核心学习信号，驱动持续向前运动。
  2. `survival_height` – 生存约束，避免因高度越界终止。
  3. `stay_upright` – 姿态稳定，防止侧翻导致的连锁失败。
  4. `suppress_lateral_velocity` – 抑制侧滑，提高能量效率和步态质量。

## 信号到公式算子的映射
| Reward role | Signal | Formula operator | 设计理由 |
|---|---|---|---|
| forward_progress | `next_obs[13]` (body_x_velocity) | **bounded signal**（`vx/(1+vx)`） | 有界单调递增，低速度时梯度充足，高速度时饱和，避免“冲刺跌倒”的暴利刷分。 |
| survival_height | `next_obs[0]` (body_z) | **dense_state_signal (hinge)** | 只在接近终止边界（<0.3 或 >0.9）时生效，安全区内不产生无谓惩罚，给机器人更大的探索空间。 |
| stay_upright | `body_up_z` (由四元数计算) | **quadratic_penalty** (error²) | 连续惩罚偏离完全直立的状态，梯子始终存在，且权重不大到压制前进动力。 |
| suppress_lateral_velocity | `next_obs[14]` (body_y_velocity) | **dense_state_signal (绝对值)** | 轻量惩罚侧滑，帮助机器人形成前后对称的步态，避免横向漂移。 |

## 排除的职责与原因
- **terminal_success_reward / terminal_failure_penalty**：`explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，info 为空，无法可靠使用。
- **动作平滑 / 能量惩罚**：v1 阶段优先搞清任务方向，过强的动作代价曾导致分数崩溃（-55 到 -353），留到后续迭代。
- **高度奖励（正反馈）**：之前的全时高度奖励可能诱导机器人跳起或误认为“越高越好”；改用 hinge 后仅在危险区干预。
- **复杂门控 / 动态课程**：历史尝试中 gated 结构全部严重负分，v1 避免引入高阶耦合。

## 为什么这个结构能突破停滞
1. **bounded forward** 从根本上改变了主梯度的形态：它在低速区提供强梯度，在高速区自限，对抗“为速度牺牲稳定性”的捷径。
2. **hinge height penalty** 替换了之前的全时高度奖励/惩罚，让机器人在安全区内不受高度约束干扰，能自由探索不同的步态姿态。
3. **独立 upright penalty** 保持了对直立的直接引导，而非像 gated 结构那样完全依赖 forward 的乘积放大，避免了“倾斜后梯度消失”的死局。
4. 整体尺度设计使得正常行走（vx≈0.3, z≈0.5, up≈1, vy≈0）时总奖励 > 0，避免了负奖励为主的抑制效应（前次 pure‑penalty 方案得 -5.09）。

## 后续迭代预留
- 若生存率提高但速度停滞，可将 forward 改为凸化形式 (`vx²` 或 `exp`)。
- 若关节抖动严重，可加入轻量 **动作变化率惩罚**（需利用 `action` 与上一动作的差，但目前无历史存储，需环境支持）。
- 若 agent 开始利用“原地踏步”骗 forward 奖励，可引入 **里程近似**（但当前无位置信息，需借助速度积分或环境扩展）。

## 训练后应观察的失败模式
- **高度塌缩**：hinge 阈值设得太宽（0.3）可能导致 agent 长时间在过低高度行走而未被充分警告，需监控最低 body_z 分布。
- **侧倾但未倒下**：upright 惩罚权重不够可能让机器人学会侧身前进，应检查 body_up_z 的 episode 均值。
- **跳动前进**：bounded forward 虽然抑制了极端速度，但 agent 仍可通过高频弹跳短暂获得正向速度，需观察 body_z 的时序波动。
- **速度饱和后停滞**：当 forward_reward 接近 1.0 后，进一步加速几乎无额外收益，agent 可能不再冒险提高速度；若想突破速度上限，可后续引入 `vx**2` 或指数形式。
