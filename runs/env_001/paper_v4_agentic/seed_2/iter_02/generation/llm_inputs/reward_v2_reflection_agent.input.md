# 1. Search objective
- target_score: 200.000000
- current_score: -116.462541
- gap_to_target: 316.462541
- target_achievement_ratio: -58.231%

# 2. 上一轮奖励函数代码（该轮得分: -116.462541）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for the 2D lander soft‑landing task.
    Components:
      - proximity_delta:  improvement in distance to target (core driving signal)
      - velocity_penalty: quadratic speed penalty gated by proximity (soft landing)
      - orientation_penalty: quadratic penalty on tilt and angular rate (safety)
    """
    # ── current state ──
    x_cur = obs[0]
    y_cur = obs[1]
    vx_cur = obs[2]
    vy_cur = obs[3]
    angle_cur = obs[4]
    angvel_cur = obs[5]

    # ── next state ──
    x_next = next_obs[0]
    y_next = next_obs[1]

    # ── distance to pad (target at 0, 0) ──
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # ── weights / thresholds ──
    w_prox = 1.0           # core progression weight
    w_vel  = 0.5           # velocity penalty weight
    w_ang  = 0.1           # orientation penalty weight
    proximity_threshold = 0.5   # distance below which we start caring about speed

    # ── 1. Proximity delta (improvement_delta) ──
    # positive when the lander reduces its distance to the target.
    proximity_delta = w_prox * (dist_cur - dist_next)

    # ── 2. Soft landing velocity penalty (quadratic_penalty + conditional_gating) ──
    # gate: 0 when far away, linear ramp to 1 when inside threshold.
    gate = max(0.0, 1.0 - dist_cur / proximity_threshold)
    speed_sq = vx_cur ** 2 + vy_cur ** 2
    velocity_penalty = -w_vel * speed_sq * gate

    # ── 3. Orientation stability (quadratic_penalty) ──
    orientation_penalty = -w_ang * (angle_cur ** 2 + angvel_cur ** 2)

    # ── Total reward ──
    total_reward = proximity_delta + velocity_penalty + orientation_penalty

    components = {
        "proximity_delta": proximity_delta,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-116.462541, len=134.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-154.656939, -66.012571]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_delta | 0.872753 | 83.2% | 91.4% | 100.0% |
| velocity_penalty | -0.060163 | -5.7% | 5.7% | 1.5% |
| orientation_penalty | -0.030161 | -2.9% | 2.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 12/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: mean_eval_score=-116.46, 12/20 episodes terminate early (<150 steps), all 20/20 terminated. Per-step generated_reward total=0.0002 vs original_env_reward=-0.7575 — the shaped reward is ~3800× smaller than the environment penalty, making it invisible as a learning signal. proximity_delta dominates generated components at 91.4% magnitude share but adds only ~0.005/step.

**Component Anomalies**: velocity_penalty active_rate=1.5% — nearly dead. The gate (threshold=0.5, dist_cur computed from origin) almost never opens because the lander rarely reaches within 0.5 of the pad. orientation_penalty is always active (100%) but contributes only 2.9% magnitude share with negligible per-step magnitude.

**Mechanism Hypothesis**: The generated reward components are dwarfed by the raw environment penalty, so the policy receives essentially no shaped guidance. The velocity_penalty gate is excessively strict (dist_cur < 0.5), rendering it a dead component that cannot shape approach behavior or encourage soft landing during descent.

**Decision Implication**: PATCH: (1) scale all generated components up by at least 100× so they are comparable to original_env_reward magnitude; (2) relax the velocity_penalty proximity gate (e.g., threshold ≥5.0 or use a smooth falloff) so it can shape the full descent trajectory, not just the final 0.5 units.

**Confidence**: `high`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
任务主目标是控制一台具有左右支撑腿和主/侧向引擎的 2D 着陆器，从视野顶部中央附近以随机初始推力开始，尽快到达场地中央的着陆垫，并稳定、安全地停靠在该垫上。次要目标是尽量减少引擎使用（燃料消耗），同时保持车身姿态稳定，实现软着陆。不应将单纯的快速到达或单纯省燃料作为独立核心目标，代理必须在安全着陆的前提下兼顾速度与能效。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（根据环境惯例，具体从环境读取，但推测为 float）  
- obs[0]: x_position，相对目标垫的水平坐标，reward_usable: true  
- obs[1]: y_position，相对着陆垫高度的垂直坐标，reward_usable: true  
- obs[2]: x_velocity，水平线速度，reward_usable: true  
- obs[3]: y_velocity，垂直线速度，reward_usable: true  
- obs[4]: body_angle，车身俯仰/横滚角度，reward_usable: true  
- obs[5]: angular_velocity，角速度，reward_usable: true  
- obs[6]: left_support_contact，左支撑腿接触标志（0.0 或 1.0），reward_usable: true（需谨慎使用）  
- obs[7]: right_support_contact，右支撑腿接触标志（0.0 或 1.0），reward_usable: true（需谨慎使用）

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine，不点火，不做任何事情  
- action 1: left_orientation_engine，点燃左姿态引擎（产生方向性的力）  
- action 2: main_engine，点燃主引擎（通常向上推力）  
- action 3: right_orientation_engine，点燃右姿态引擎（与左引擎反向）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 当 `body_not_awake_or_settled` 为真，且未发生 crash/出界时，可推断为成功着陆并稳定。
- failure-like termination: `crash_or_body_contact`（猛烈碰撞或非法身体接触）和 `horizontal_position_outside_viewport`（水平飞出视野）均视为失败。
- ambiguous termination: 单独的 `body_not_awake_or_settled` 可能发生在成功着陆后不久，也可能是因摔落后静止，需结合位置、速度判断。
- truncation: 环境中未提供 truncation 信号（step 返回 `terminated, False`），因此不存在时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（step 返回 info={}）  
- forbidden_or_uncertain_info_fields: 任何 info 字典中的字段均不存在，不可使用

间接推断路径：
- 成功着陆可从 termination 时接近目标垫中心、垂直速度极低、角度接近 0 且至少一腿接触的条件组合推断（derived_possible）。
- crash 可从 termination 时 y_position 骤降至地面以下或水平位置远超边界等条件间接检测（derived_possible）。

## 7. 可用于奖励函数的信号
- position：
  - 相对垫的水平距离 `|x_position|`
  - 相对垫高度的垂直距离 `|y_position|`（当 y_position 为正代表高于垫，到达垫面时理想 y≈0）
- velocity：
  - 水平速度 `x_velocity`（软着陆要求接近0）
  - 垂直速度 `y_velocity`（负值代表下落，着陆瞬间需要小）
- orientation：
  - body_angle（理想接近0，可取其绝对值或二次惩罚）
  - angular_velocity（软着陆应接近0）
- contact：
  - left_support_contact / right_support_contact（至少一腿接触可能表明着陆成功，但需结合速度和位置，否则可能鼓励猛烈砸地）
- action/engine：
  - action 本身（可计算引擎使用惩罚，no_engine 时无推力）
- other：
  - 通过 next_obs 与 obs 的差值得出速度变化，可用于检测剧烈推力
  - 衍生信号：是否接近目标垫且速度降低（derived_possible）

# 7. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | orientation_penalty + proximity_delta + velocity_penalty | -116.46 | -116.46 | 0.00 | 134.15 | orientation_penalty=-0.003 proximity_delta=0.005 velocity_penalty=-0.002 | new_best |
