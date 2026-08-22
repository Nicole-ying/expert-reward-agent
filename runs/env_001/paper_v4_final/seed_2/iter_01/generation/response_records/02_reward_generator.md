# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs dimensions per environment card
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Primary progress signal: dense quadratic penalty on position and velocity errors.
    #    Encourages the lander to move toward (0,0) with zero speed.
    pos_sq_error = x_pos**2 + y_pos**2
    vel_sq_error = x_vel**2 + y_vel**2
    progress = -0.05 * pos_sq_error - 0.1 * vel_sq_error

    # 2. Stability constraint: quadratic penalty on body angle and angular velocity.
    #    Keeps the lander upright and prevents excessive spinning.
    pose_penalty = -5.0 * (body_angle**2) - 0.5 * (angular_vel**2)

    # 3. Soft landing bonus: a task-completion proxy active when both legs are grounded.
    #    The bonus is large only when touchdown is gentle (low speed, nearly vertical).
    both_leg_grounded = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    speed_magnitude = abs(x_vel) + abs(y_vel)
    speed_factor = 1.0 / (1.0 + 5.0 * speed_magnitude)        # bounded signal: decays with speed
    angle_factor = 1.0 / (1.0 + 20.0 * abs(body_angle))      # bounded signal: decays with tilt
    landing_bonus = 10.0 * both_leg_grounded * speed_factor * angle_factor

    total_reward = progress + pose_penalty + landing_bonus

    components = {
        'progress': progress,
        'pose_penalty': pose_penalty,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **2D 着陆器导航目标到达**（goal-reaching with soft-landing constraints）。
- 动力学亚型：离散推力控制（4 个动作），机身受重力、引擎力和力矩影响，终止条件包括坠毁、出界和稳定着陆。

## selected reward roles
1. **主进展信号（progress）** — 必须包含，驱动 agent 尽快逼近目标位置并减速至零。
2. **稳定性约束（pose_penalty）** — 健康/安全角色，防止机身倾角和角速度过大。
3. **软着陆奖励（landing_bonus）** — 任务完成近似信号，在双腿同时接地时给予连续软着陆评价。

## role_to_signal_mapping
| 角色 | 使用的观测信号 (next_obs 索引) | 公式算子 |
|------|-------------------------------|----------|
| progress | `[0] x`, `[1] y`, `[2] vx`, `[3] vy` | dense_state_signal (quadratic penalty) |
| pose_penalty | `[4] body_angle`, `[5] ang_vel` | quadratic_penalty |
| landing_bonus | `[6] left_contact`, `[7] right_contact`, 速度, 角度 | joint_condition_proxy（乘积）+ bounded_signal（1/(1+k·error)） |

## formula operator choices
- **progress**: `-w_pos * (x² + y²) - w_vel * (vx² + vy²)` — dense 二次惩罚，每步都有梯度；
- **pose_penalty**: `-w_angle * angle² - w_angvel * angvel²` — 二次惩罚，无硬边界，允许必要的微小倾斜；
- **landing_bonus**: 由 `both_leg_grounded * speed_factor * angle_factor` 构成，speed_factor/angle_factor 使用 `1/(1 + k*error)` 类型 bounded_signal，在着陆瞬间给予连续评价，避免乘积塌缩（因子在安全范围内接近 1）。

## excluded roles 及原因
- **efficiency / 动作代价** — 环境虽有“节省燃料”目标，但 v1 阶段主任务尚未收敛，动作代价留待后续加入；
- **terminal_success_reward / terminal_failure_penalty** — environment_card 声明 `explicit_success_flag_available=false`、`explicit_failure_flag_available=false`，info 字典为空，无法可靠判断终止类型；
- **preview_conditioned_reward** — 观测空间未提供前方地形或距离传感器信号；
- **action_smoothness_penalty** — 无 previous action 历史，离散动作空间也无需此项。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境未提供显式成功/失败标志，`info` 为空，终止类型（crash、out‑of‑viewport、settled）无法直接获取。因此改用：
- **per‑step 的负向进展惩罚** 持续推动 agent 向目标靠近；
- **软着陆奖励** 在双腿触地且姿态、速度良好时给予正向引导，作为任务完成的近似信号；
- 摔毁或飞出的不良行为通过**主惩罚累积和姿势约束**被隐式惩罚（若 agent 倾覆或高速撞击，其 `pose_penalty` 和 `progress` 都会恶化）。

## 哪些职责留到后续迭代
- 主引擎/姿态引擎的**离散动作效率代价**（燃料节省）；
- 更精细的**动态课程**（例如根据训练进度收紧速度/角度容忍阈值）；
- 若后续观测中出现前方测距信息，可加入**preview 门控**。

## 训练后应该观察哪些 failure modes
1. **高速坠毁**：若 `landing_bonus` 权重不足以抵消快速终止带来的累积惩罚减少，agent 可能选择直冲地面提前结束回合。应观察 landing_bonus 与 progress 惩罚在末尾步的数值对比。
2. **悬停静止**：位置惩罚持续存在，agent 可能学会缓慢漂移以减少每步速度惩罚，导致回合冗长且无法着陆。需检查平均 `y_vel` 是否长期接近于 0 而 `y_pos` 下降缓慢。
3. **过度倾斜移动**：agent 可能利用大角度产生水平推力而不触发终止，`pose_penalty` 若过轻则倾斜频繁；若过重则可能不敢校正水平位置。需要监控 `body_angle` 的均值与方差。
4. **着陆后弹跳/腿擦地奖励黑客**：由于着陆奖励仅在双腿接触时激活，但环境在稳定着陆后即终止，无法重复获取；若平台边缘允许单腿触地后弹起，agent 可能尝试反复接触。应分析 `landing_bonus` 的 active rate 是否异常增高。

