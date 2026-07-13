# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Hyperparameters
    w_progress = 1.5
    k_landing = 4.0
    w_contact = 0.3
    w_speed_landing = 0.2
    w_angle = 0.5
    w_global_speed = 0.01

    # Current and next distances to target pad (relative coordinates)
    dist_now = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # --- Component A: progress toward target ---
    progress_reward = w_progress * (dist_now - dist_next)

    # --- Component B: soft‑landing (activated by proximity) ---
    landing_factor = 1.0 / (1.0 + k_landing * dist_next)
    contact_bonus = (next_obs[6] + next_obs[7]) * w_contact
    speed_penalty = w_speed_landing * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    soft_landing = landing_factor * (contact_bonus - speed_penalty - angle_penalty)

    # --- Component C: gentle global speed suppression ---
    global_speed_penalty = -w_global_speed * (abs(next_obs[2]) + abs(next_obs[3]))

    total_reward = progress_reward + soft_landing + global_speed_penalty
    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "global_speed_penalty": global_speed_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **任务类型**：`navigation_goal_reaching`，子类型 `goal_approach_and_soft_contact`；飞行器需从上方降落到目标垫并稳定静止。
- **选定的奖励职责**：
  1. **`approach_progress`**（强制）—— 使用 `improvement_delta` 算子，鼓励每步缩短到目标垫的欧氏距离。
  2. **`soft_landing`**（强制）—— 在接近目标垫时激活（通过 `bounded_signal` 系数 `1/(1+k*dist_next)`），内部组合了接触奖励、速度惩罚和姿态惩罚，引导慢速、低角度、双腿触地着陆。
  3. **`global_speed_penalty`**（条件式轻量约束）—— 每步极小幅度的速度惩罚，为 agent 提供“减速趋向”的微弱梯度，避免过度高速撞击目标垫，但不压制探索。
- **信号映射**：
  - `approach_progress` 用 `obs[0,1]` 和 `next_obs[0,1]` 计算距离变化。
  - `soft_landing` 用 `next_obs[0..7]` 中的距离、接触、速度、角度。
  - `global_speed_penalty` 用 `next_obs[2,3]` 的速度。
- **公式算子选择**：
  - `approach_progress` → `improvement_delta`
  - `soft_landing` 中的距离门 → `bounded_signal`（倒数形式），接触奖励为线性正项，速度/角度惩罚为二次或线性（这里用绝对值近似二次平滑效果）
  - `global_speed_penalty` → `dense_state_signal` 的线性惩罚变体
- **排除的职责及原因**：
  - `terminal_success_reward`：无显式成功标志，`info` 为空，无法可靠区分成功/失败。
  - `crash_penalty_based_on_info`：无碰撞信息。
  - `fuel_penalty`：v1 阶段避免压制探索，且历史尝试中燃油惩罚组合得分极差，留到后续迭代。
  - `orientation_penalty` 作为独立角色已并入 `soft_landing`。
- **未使用的 term**：`terminal_success_reward`、`terminal_failure_penalty` 因信号缺失完全弃用；未引入任何门控训练课程或动态权重。
- **后续迭代方向**：若出现高速撞击但距离很近的失败模式，可将 `global_speed_penalty` 升级为与距离相关的 hinge 形式；若体能学会慢速接触但仍无法稳定，可加强 `soft_landing` 中接触不对称惩罚或角速度约束；燃油优化可后续较轻权重加入。
- **期待观察的失败模式**：
  - 只悬停不下降（检查 progress 是否下降缓慢 → 增大 `w_progress` 或减小速度惩罚）
  - 高速撞垫（检查 `global_speed_penalty` 是否足够，可能需加大权重或转为基于距离的 gate）
  - 在目标垫上方摇摆（检查 landing 区域的速度/角度惩罚是否及时激活）
