# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract observation indices ----------
    body_z_next = next_obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_vel = obs[13]
    body_y_vel = obs[14]

    # ---------- forward velocity reward (primary) ----------
    w_fwd = 1.5
    forward_reward = w_fwd * body_x_vel

    # ---------- body height safety hinge ----------
    # punish when height is approaching the low termination boundary (0.2)
    safe_low = 0.3
    height_err = safe_low - body_z_next
    w_height = 100.0
    height_penalty = -w_height * max(0.0, height_err) ** 2

    # ---------- upright orientation safety hinge ----------
    # body_up = 1 - 2*(qx^2 + qy^2), range [-1, 1]
    body_up = 1.0 - 2.0 * (qx * qx + qy * qy)
    unsafe_up = 0.5
    up_err = unsafe_up - body_up
    w_up = 5.0
    upright_penalty = -w_up * max(0.0, up_err) ** 2

    # ---------- lateral drift penalty ----------
    w_lat = 0.5
    lateral_penalty = -w_lat * (body_y_vel ** 2)

    # ---------- total reward ----------
    total_reward = forward_reward + height_penalty + upright_penalty + lateral_penalty

    components = {
        "forward_reward": forward_reward,
        "height_penalty": height_penalty,
        "upright_penalty": upright_penalty,
        "lateral_penalty": lateral_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与动态子类型
- **task_family**: `locomotion_continuous_control`
- **dynamics_subtype**: `multi_legged_body_locomotion`  
  四足机器人在扭矩控制下需要产生持续前进运动，同时维持身体高度和躯干直立。

## 2. 选中的奖励职责 (Selected Reward Roles)
| 角色 | 信号 | 公式算子 | 设计理由 |
|------|------|----------|----------|
| `forward_velocity_reward` (主学习信号) | `obs[13]` (body_x_velocity) | `dense_state_signal (linear positive)` | 直接给予前进速度线性奖励，每步提供稠密梯度，驱动 agent 产生有效位移 |
| `height_survival_reward` (安全约束) | `next_obs[0]` (body_z) | `dense_state_signal (hinge penalty)` | 仅在身体高度接近危险低边界（0.3 m）时给予二次惩罚，避免摔倒；在安全范围（>0.3）内无惩罚，不压制正常探索 |
| `upright_orientation_reward` (安全约束) | `obs[2], obs[3]` (quat_x, quat_y) 计算 `body_up` | `dense_state_signal (hinge penalty)` | 惩罚躯干过度倾斜（`body_up < 0.5`），防止侧翻，保持步态稳定；容忍小幅倾斜以保证自然步态 |
| `lateral_motion_penalty` (方向约束) | `obs[14]` (body_y_velocity) | `quadratic_penalty` | 轻量惩罚侧向漂移，使前进方向更纯净，但权重很小，不影响对称步态的微小侧摆 |

**共 4 个组件**，覆盖主驱动力、两个生存约束、一个效率/方向约束，符合 v1 阶段 2–4 个组件的预算。

## 3. 未选中的角色与原因 (Excluded Roles)
- **`action_magnitude_penalty`**: 早期加入会压制探索，历史试验显示加入后得分明显为负，说明 agent 尚未掌握前进就被限制力矩，留到后续迭代。
- **`joint_velocity_penalty`**: 类似理由，过早抑制动作多样性不利于形成步态。
- **`soft_health_gate`**: 虽然具有理论优势，但此前尝试（`forward_gated_height` 等组合）全部给出负分，说明在当前任务上乘积式 gate 可能过早切断微弱的前进信号，使学习停滞。本版改用独立 hinge 惩罚，只在安全边界附近介入。
- **`terminal_success_reward`**: 环境不存在显式成功标志 (`explicit_success_flag_available=false`)，且没有可用的 info 字段，无法实现。
- **`terminal_failure_penalty`**: 奖励函数无权访问 terminated 标志，只能通过 `next_obs[0]` 预测危险并提前惩罚。
- **`contact_foot_reward` / `distance_from_origin`**: 均因缺少观测信号而排除。

## 4. 关键设计决策
1. **Hinge 惩罚优于全域二次惩罚**  
   高度和直立惩罚仅在接近危险边界时激活，避免在全任务过程中持续压制主学习信号。这种设计允许 agent 在安全范围内自由调整姿态以生成步态，同时保留了在危险情况下的梯度引导。
2. **使用 `next_obs[0]` 进行高度惩罚**  
   选择下一步的高度作为惩罚输入，使 agent 提前一步感知即将发生的摔倒，比使用当前高度具有更强的前瞻性。
3. **凸化惩罚（平方）**  
   对高度误差和上身倾角误差取平方，使危险程度越高时惩罚增加得越快，形成“软急停”效果。
4. **权重尺度校准**  
   - `w_fwd=1.5` 使前进奖励（正常行走时约 1–3）成为主要收益。
   - `w_height=100` 使高度降至 0.25 m 时惩罚约 -0.25，降至 0.22 m 时约 -0.64，与前进奖励可比，安全区内为零。
   - `w_up=5` 使上身倾角过大时产生适度惩罚（例如 `body_up=0.3` 时惩罚 -0.2），严重翻转时惩罚迅速增大。
   - `w_lat=0.5` 使侧向速度惩罚保持在很小数量级（0.01–0.05），不干扰主学习。

## 5. 后续迭代方向
- 待前进能力稳定后，可引入 **动作幅度惩罚** 或 **平滑性惩罚**（若环境提供历史动作）以提高能效。
- 如果后期出现“冲刺‑摔倒”反复，可重新评估 **soft_health_gate** 的使用方式，但当前独立 hinge 更稳妥。
- 可尝试 **joint_condition_proxy** 来组合多个安全条件，给接近完美直立且高速移动的状态额外奖励，替代显式成功信号。

## 6. 预期需观察的失败模式
- **站立不动**：若 agent 发现前进会导致摔倒而选择静止，`forward_reward` 会为 0，但 hinge 惩罚也为 0（只要不动时高度和直立正常），可能陷入零速度策略。可通过轻微提高 `w_fwd` 或加入极小常数生存奖励（未在此版实施）缓解。
- **低速振荡**：如果高度惩罚过于敏感，agent 可能在 0.3 米附近高频调节步态，导致效率低下。应监控 `height_penalty` 的步均值与触发频率。
- **侧向自旋**：虽然加入 `lateral_penalty`，但若步态固有大幅侧摆，可能被过度惩罚，需观察侧向速度和前进速度的比例，必要时降低权重。
- **高跳近上限**：未设置上边界惩罚，若 agent 学会跳跃式前进而接近高度上限（1.0 m）可能触发终止，后续需增加上边界 hinge。
