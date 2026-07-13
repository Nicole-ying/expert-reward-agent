# ⚠️ 上一版代码验证失败
错误信息：Reward v9 failed validation: 缺少准确函数签名; 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。; 未发现明显的 total_reward/reward 变量名 (record: runs\env_001\paper_v4\seed_0\iter_09\generation\validations\reward_v9.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="search_reward_design_knowledge">
<｜｜DSML｜｜parameter name="query" string="true">state-based proximity reward dominating 70% share, agent landing successfully but accumulating too much proxy, should convert to improvement-based or reduce scale</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>

# Search objective
- target_score: 200.000000
- current_score: 224.208876
- gap_to_target: -24.208876
- target_achievement_ratio: 112.104%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 224.208876）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous state (needed for touchdown transition detection)
    prev_lc = obs[6]
    prev_rc = obs[7]

    # Next state
    x = next_obs[0]
    y = next_obs[1]
    xv = next_obs[2]
    yv = next_obs[3]
    ang = next_obs[4]
    lc = next_obs[6]
    rc = next_obs[7]

    # 1. Landing proximity: state-based reward for horizontal centering AND low height
    horizontal_proximity = 1.0 / (1.0 + 5.0 * abs(x))
    height_factor = max(0.0, 1.0 - y / 2.0)
    landing_proximity = 2.0 * horizontal_proximity * height_factor

    # 2. Descent safety: penalize dangerous high-speed descent near ground
    descent_safety = 0.0
    if yv < 0.0 and y > 0.02:
        total_speed = (xv ** 2 + yv ** 2) ** 0.5
        height_urgency = 1.0 / (1.0 + y)
        descent_safety = -2.0 * total_speed * height_urgency

    # 3. Touchdown bonus: first moment both legs achieve ground contact
    prev_contacts = prev_lc + prev_rc
    next_contacts = lc + rc
    touchdown_bonus = 0.0
    if next_contacts >= 2.0 and prev_contacts < 2.0:
        speed_quality = max(0.0, 1.0 - (abs(xv) + abs(yv)))
        angle_quality = max(0.0, 1.0 - abs(ang) / 0.5)
        touchdown_bonus = 3.0 * (1.0 + speed_quality * angle_quality)

    # 4. Stable landed: ongoing reward for maintaining a successful landing state
    stable_landed = 0.0
    if lc > 0.5 and rc > 0.5 and y < 0.3:
        speed_score = max(0.0, 1.0 - (abs(xv) + abs(yv)))
        angle_score = max(0.0, 1.0 - abs(ang))
        stable_landed = speed_score * angle_score

    # 5. Fuel penalty: small cost for any engine use
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = landing_proximity + descent_safety + touchdown_bonus + stable_landed + fuel_penalty

    components = {
        "landing_proximity": landing_proximity,
        "descent_safety": descent_safety,
        "touchdown_bonus": touchdown_bonus,
        "stable_landed": stable_landed,
        "fuel_penalty": fuel_penalty
    }

    return (float(total_reward), components)
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=224.208876, len=437.100000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[18.631804, 279.104121]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proximity | 415.061881 | 70.3% | 70.3% | 100.0% |
| descent_safety | -91.659276 | -15.5% | 15.5% | 77.6% |
| stable_landed | 50.435412 | 8.5% | 8.5% | 13.2% |
| fuel_penalty | -18.005000 | -3.0% | 3.0% | 82.4% |
| touchdown_bonus | 15.247017 | 2.6% | 2.6% | 0.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主要目标：控制一个二维刚体从上方初始位置尽快到达并稳定停靠在视野中央的目标垫上。  
次要目标：在满足成功停靠的前提下，尽可能少地使用发动机推力（节约燃料/能量）。  
不允许混淆的目标：这是一个典型的到达＋软着陆任务，核心不是维持平衡或长时间存活，而是以安全、低能耗的方式精确到达指定位置。快速性隐含在“as fast as possible”中，但需要与能耗目标协调。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断，前6维连续，后2维取0/1但通常保持浮点）
- obs[0]: x_position，相对于目标垫中心的水平距离，reward_usable: true
- obs[1]: y_position，相对于垫面的垂直距离（高度），reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，刚体俯仰角（与垂直/水平方向的偏差），reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿触地标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右支撑腿触地标志（0/1），reward_usable: true

所有8维均可直接用于奖励计算（位移、速度、姿态、接触状态）。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不施加任何推力，惯性飞行
- action 1: left_orientation_engine — 点燃一侧方向引擎（可能产生侧向力矩和/或力，用于调整姿态）
- action 2: main_engine — 点燃主引擎（产生向上的推力，同时可能引起力矩变化）
- action 3: right_orientation_engine — 点燃另一侧方向引擎（与左引擎相反方向）

注意：动作空间是符号化的，但每个动作对“燃料消耗”有隐含影响（0号动作不消耗燃料）。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（“settled”指身体静止并稳定地站立在垫上）是可能的成功终止；但仅凭该条件无法绝对区分成败，因为也可能包含非成功静止（如横躺在地面但物理不活动）。
- failure-like termination: `crash_or_body_contact`（主体部分与地面或其它障碍激烈碰撞）、`horizontal_position_outside_viewport`（偏离目标太远）。
- ambiguous termination: 所有的终止条件均不显式标记成功/失败，需结合观测（位置、速度、姿态、双腿触地）判断。
- truncation: 源未提供步数截断信息，假设环境无显式步数限制，但从实际训练安全考虑通常会有时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空字典，无可用字段）
- forbidden_or_uncertain_info_fields: 没有任何 info 字段被允许供奖励使用，因此不能依赖任何隐藏在 info 中的成功标志。

## 7. 可用于奖励函数的信号
- position: 相对垫子的水平距离 x_position (obs[0])，高度 y_position (obs[1])
- velocity: 水平速度 x_velocity (obs[2])，垂直速度 y_velocity (obs[3])
- orientation: 身体角度 body_angle (obs[4])，角速度 angular_velocity (obs[5])
- contact: 左腿触碰 left_support_contact (obs[6])，右腿触碰 right_support_contact (obs[7])
- action/engine: 动作类型 (action)，可以推断燃料消耗（非零动作消耗）
- other: 从 next_obs 可计算速度变化、当前信息；终止可与接触模式结合判断着陆是否完全稳定。